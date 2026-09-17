use super::{
    Arc, Environments, Index, Inputs, InstallOptions, Interpreter, IntoDiagnostic, Lockfile,
    PYPI_ECOSYSTEM, Path, PathBuf, PnprClient, PypiResolveOptions, Result, StoreIndexWriter, bail,
    fs, io, manifest,
};
use miette::WrapErr;
use std::collections::{BTreeMap, BTreeSet};

/// What every project of one [`prepare`](super::prepare) run shares,
/// before the interpreter each one is installed with is known.
pub(super) struct Shared<'a> {
    pub(super) context: &'a InstallOptions,
    pub(super) index: &'a Index,
    pub(super) store: ArtifactStore<'a>,
    pub(super) asked: Asked,
    /// Every distribution a project in this repository declares. A build
    /// requirement naming one of them is refused rather than taken from
    /// the index, wherever the backend asked for it.
    pub(super) members: BTreeMap<std::path::PathBuf, BTreeSet<pep508_rs::PackageName>>,
    pub(super) caches: Caches,
}

#[derive(Default)]
pub(super) struct Caches {
    /// The environments backends have already been installed into. Every
    /// project using one backend needs the same environment, and a
    /// workspace is mostly one backend.
    pub(super) build_environments: BuildEnvironments,
    pub(super) resolutions: super::resolutions::Resolutions,
}

/// A backend environment belongs to the interpreter that installed it and
/// the requirements it holds: a backend runs in the interpreter it was
/// installed for, and what it compiles is built for that one.
pub(super) type BuildEnvironmentKey = (String, String);

type BuildEnvironmentEntry = Arc<tokio::sync::Mutex<Option<Arc<tempfile::TempDir>>>>;

pub(super) type BuildEnvironments =
    tokio::sync::Mutex<BTreeMap<BuildEnvironmentKey, BuildEnvironmentEntry>>;

/// What preparing one project needs: what the run shares, plus the
/// interpreter that installs this project and the environments it is
/// locked for.
#[derive(Clone)]
pub(super) struct PythonPrepare<'a> {
    pub(super) context: &'a InstallOptions,
    pub(super) interpreter: &'a Interpreter,
    pub(super) environments: &'a Environments,
    pub(super) index: &'a Index,
    pub(super) store: ArtifactStore<'a>,
    pub(super) asked: Asked,
    pub(super) members: &'a BTreeMap<std::path::PathBuf, BTreeSet<pep508_rs::PackageName>>,
    pub(super) state: PreparationState<'a>,
}

#[derive(Clone)]
pub(super) struct PreparationState<'a> {
    pub(super) caches: &'a Caches,
    pub(super) building: BTreeSet<BuildEnvironmentKey>,
}

impl<'a> PythonPrepare<'a> {
    pub(super) fn for_project(
        shared: &'a Shared<'a>,
        interpreter: &'a Interpreter,
        environments: &'a Environments,
    ) -> Self {
        Self {
            context: shared.context,
            interpreter,
            environments,
            index: shared.index,
            store: ArtifactStore { index: shared.store.index.clone(), writer: shared.store.writer },
            asked: shared.asked,
            members: &shared.members,
            state: PreparationState { caches: &shared.caches, building: BTreeSet::new() },
        }
    }
}

/// What the install asked this run for.
#[derive(Clone, Copy)]
pub(super) struct Asked {
    /// Whether a dependency is being added, which resolves again.
    pub(super) resolve: bool,
    pub(super) selection: manifest::DependencySelection,
}

/// The store a run reads verified artifacts from and writes them to.
#[derive(Clone)]
pub(super) struct ArtifactStore<'a> {
    pub(super) index: Option<pnpm_store_dir::SharedReadonlyStoreIndex>,
    pub(super) writer: &'a Arc<StoreIndexWriter>,
}

/// What [`PythonPrepare::lockfile`] needs about one project.
pub(super) struct LockfileInputs<'a> {
    /// The lockfile on disk, when it still applies to these inputs on this
    /// target.
    pub(super) existing: Option<Lockfile>,
    pub(super) lock_path: &'a Path,
    pub(super) requirements: &'a [pep508_rs::Requirement],
    pub(super) inputs: Inputs,
    pub(super) requires_python: Option<String>,
    /// The projects in this repository the resolution installs from their
    /// source, which every seeding of it has to offer again.
    pub(super) local: Arc<[super::workspace::LocalProject]>,
    /// The projects the requirements were collected from, which a failed
    /// resolution is explained in terms of.
    pub(super) members: &'a [super::projects::Member],
}

/// What [`PythonPrepare::replay_lockfile`] needs about the lockfile on
/// disk.
pub(super) struct LockfileReplay<'a> {
    pub(super) lock: Lockfile,
    pub(super) lock_path: &'a Path,
    pub(super) requirements: &'a [pep508_rs::Requirement],
    pub(super) local: Arc<[super::workspace::LocalProject]>,
    /// Whether the lockfile was resolved for this install's own target,
    /// so that every wheel it pins is one this target needs.
    pub(super) same_target: bool,
}

/// The lockfile beside the project, or `None` when it has none yet.
pub(super) async fn read_existing_lock(lock_path: &Path) -> Result<Option<Lockfile>> {
    let contents = match tokio::fs::read_to_string(lock_path).await {
        Ok(contents) => contents,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).into_diagnostic(),
    };
    toml::from_str::<Lockfile>(&contents)
        .into_diagnostic()
        .wrap_err_with(|| format!("parse {}", lock_path.display()))
        .map(Some)
}

/// Refuse a lockfile that answers a different question than this install
/// asked. Writing one would leave behind a lockfile the next install reads
/// back as stale, and a frozen install would fail on it outright.
pub(super) fn accept_server_lockfile(
    lock: &Lockfile,
    inputs: &Inputs,
    requires_python: Option<&str>,
) -> Result<()> {
    if lock.tool.pnpm != *inputs || lock.requires_python.as_deref() != requires_python {
        bail!("the pnpr server resolved Python dependencies for other inputs");
    }
    Ok(())
}

/// Resolve through the configured pnpr server, which reads the index and
/// each wheel's metadata instead of making this client download wheels to
/// find out what they require.
///
/// `None` when there is no server to ask, or when the one configured
/// resolves Python not at all.
pub(super) async fn resolve_via_pnpr(
    config: &pnpm_config::Config,
    requirements: &[pep508_rs::Requirement],
    target: &pnpm_python_resolver::Target,
    index: &str,
    requires_python: Option<String>,
) -> Result<Option<Lockfile>> {
    let Some(pnpr_server) = config.pnpr_server.as_deref().filter(|_| !config.offline) else {
        return Ok(None);
    };
    let client = PnprClient::new(pnpr_server);
    if !pnpm_pnpr_client::server_resolves(&client, pnpr_server, PYPI_ECOSYSTEM)
        .await
        .wrap_err("negotiate Python resolution with the pnpr server")?
    {
        return Ok(None);
    }
    let resolved = client.resolve_pypi(PypiResolveOptions {
        requirements: requirements
            .iter()
            .map(ToString::to_string)
            .collect(),
        target: target.clone(),
        index: index.to_string(),
        requires_python,
        authorization: config.auth_headers.for_url(pnpr_server),
    })
    .await;
    match resolved {
        // A server answers from an index and the metadata published
        // beside a wheel. What needs this machine instead — an explicit
        // source, or a release whose metadata is in the wheel its source
        // distribution builds — is resolved here.
        Err(pnpm_pnpr_client::PnprClientError::Server(message))
            if message.starts_with("Python ")
                && message.ends_with(" must be resolved by the client") =>
        {
            Ok(None)
        }
        result => result
            .into_diagnostic()
            .wrap_err("resolve Python dependencies through the pnpr server")
            .map(Some),
    }
}

/// The environments pnpm manages, kept in the store rather than beside
/// their projects: one directory per project, holding one directory per
/// generation, with the project's `.venv` linking to the generation it
/// currently runs.
///
/// A repository with many Python projects therefore holds one link per
/// project rather than a generation directory each, and an environment is
/// on the store's filesystem, where the wheel files it shares with the
/// store can be cloned or hardlinked.
#[derive(Clone)]
pub(super) struct EnvironmentStore {
    root: PathBuf,
}

impl EnvironmentStore {
    pub(super) fn new(store: &pnpm_store_dir::StoreDir) -> Self {
        Self { root: store.root().join("python-envs") }
    }

    /// A fresh generation for the project at `root`, which publication
    /// makes the project's own once every participant has prepared.
    pub(super) fn new_generation(&self, root: &Path) -> Result<Generation> {
        self.validate_link(root)?;
        let directory = self.project_directory(root)?;
        fs::create_dir_all(&directory)
            .into_diagnostic()
            .wrap_err_with(|| {
                format!("create Python environment directory {}", directory.display())
            })?;
        // `.venv` links to the generation by this path, so it is made
        // absolute and physical here rather than at every reader.
        let directory = dunce::canonicalize(&directory)
            .into_diagnostic()
            .wrap_err_with(|| {
                format!("resolve Python environment directory {}", directory.display())
            })?;
        let directory = tempfile::Builder::new()
            .prefix("env-")
            .tempdir_in(directory)
            .into_diagnostic()?;
        Ok(Generation { directory, store: self.clone() })
    }

    /// The directory holding the generations of the project at `root`,
    /// named by the project's location so that two projects never share
    /// one.
    fn project_directory(&self, root: &Path) -> Result<PathBuf> {
        let root = dunce::canonicalize(root)
            .into_diagnostic()
            .wrap_err_with(|| format!("resolve Python project directory {}", root.display()))?;
        Ok(self.root.join(pnpm_crypto_hash::create_short_hash(&root.to_string_lossy())))
    }

    /// The generation the project's `.venv` currently links to, or `None`
    /// when it has no environment, or links to one that no longer exists.
    ///
    /// pnpm replaces only a link it made, so a `.venv` that is a directory,
    /// or a link to anything but a generation of pnpm's, is refused. A link
    /// into this store from another project's directory is still pnpm's: a
    /// project that has moved keeps its environment.
    pub(super) fn validate_link(&self, root: &Path) -> Result<Option<PathBuf>> {
        let link = root.join(".venv");
        match fs::symlink_metadata(&link) {
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error).into_diagnostic(),
            Ok(_) => {
                if !pnpm_fs::is_symlink_or_junction(&link).into_diagnostic()? {
                    bail!(
                        "pnpm will not replace an unmanaged Python environment: {}",
                        link.display(),
                    );
                }
                let Some(target) = link_target(root, &link)? else {
                    return Ok(None);
                };
                if !self.holds(&target)? && !holds_beside_project(root, &target)? {
                    bail!(
                        "pnpm will not replace an unmanaged Python environment: {}",
                        link.display(),
                    );
                }
                Ok(Some(target))
            }
        }
    }

    /// Whether `generation` is a generation directory of a project
    /// directory of this store.
    fn holds(&self, generation: &Path) -> Result<bool> {
        let root = match dunce::canonicalize(&self.root) {
            Ok(root) => root,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
            Err(error) => {
                return Err(error)
                    .into_diagnostic()
                    .wrap_err_with(|| {
                        format!("resolve Python environment store {}", self.root.display())
                    });
            }
        };
        Ok(generation.parent().and_then(Path::parent) == Some(root.as_path()))
    }
}

/// Where `link` leads, or `None` when it leads nowhere: a link to nothing
/// protects nothing, and the store a link led into may have been removed
/// since the environment was published.
fn link_target(root: &Path, link: &Path) -> Result<Option<PathBuf>> {
    let target = root.join(pnpm_fs::read_symlink_dir(link).into_diagnostic()?);
    match dunce::canonicalize(&target) {
        Ok(target) => Ok(Some(target)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error)
            .into_diagnostic()
            .wrap_err_with(|| {
                format!(
                    "resolve Python environment target {} for {}",
                    target.display(),
                    link.display(),
                )
            }),
    }
}

/// Whether `generation` is in the directory releases before pnpm 12.5 kept
/// a project's generations in, beside the project. An environment they
/// published is pnpm's to replace as much as one in the store.
fn holds_beside_project(root: &Path, generation: &Path) -> Result<bool> {
    match dunce::canonicalize(root.join(".pnpm/python-envs")) {
        Ok(beside) => Ok(generation.parent() == Some(beside.as_path())),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error).into_diagnostic(),
    }
}

/// A generation prepared in the store, removed with it unless publication
/// keeps it. Publication asks the store it came from whether the project's
/// `.venv` is a link pnpm may replace.
pub(super) struct Generation {
    pub(super) directory: tempfile::TempDir,
    pub(super) store: EnvironmentStore,
}

pub(super) fn publish_link(root: &Path, target: &Path) -> Result<()> {
    #[cfg(windows)]
    {
        let outcome =
            pnpm_fs::force_absolute_symlink_dir(target, &root.join(".venv")).into_diagnostic()?;
        if let Some(warning) = outcome.warning {
            bail!("{warning}");
        }
        Ok(())
    }
    #[cfg(unix)]
    {
        let temporary = tempfile::Builder::new()
            .prefix(".pnpm-python-link-")
            .tempdir_in(root)
            .into_diagnostic()?;
        let staged = temporary.path().join(".venv");
        // The link is moved up one level when published, so relative links must
        // be computed from their final location, not from the temporary directory.
        std::os::unix::fs::symlink(target, &staged).into_diagnostic()?;
        fs::rename(&staged, root.join(".venv")).into_diagnostic()
    }
}
