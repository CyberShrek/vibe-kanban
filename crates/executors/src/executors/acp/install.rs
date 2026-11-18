//! Installs ACP agents that are distributed as platform archives instead of npm
//! packages.
//!
//! The registry lists, per platform, an archive URL, an optional SHA-256 and the
//! command to run inside the extracted tree. Archives are unpacked once into
//! `<cache>/acp-agents/<id>/<version>-<platform>/` and reused on later runs.
//!
//! Extraction never writes outside the install directory: zip entries go
//! through [`zip::read::ZipFile::enclosed_name`] and tar archives through
//! [`tar::Archive::unpack`], which both reject absolute and parent-relative
//! paths.
//!
//! The registry is remote input, so archives are only fetched over HTTPS and
//! only when the registry entry carries a SHA-256. Spawning never downloads:
//! [`installed`] serves an existing install and otherwise fails, downloads
//! happen only from the settings install route.

use std::{
    collections::HashMap,
    io::{Cursor, Read},
    path::{Component, Path, PathBuf},
    sync::{Arc, LazyLock, Mutex as StdMutex},
    time::Duration,
};

use futures::StreamExt;
use serde::Serialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::executors::acp::registry::BinaryTarget;

/// Large binaries (Cursor, Junie) take a while on slow connections.
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(10 * 60);
const MAX_ARCHIVE_BYTES: u64 = 1024 * 1024 * 1024;
/// Zip bombs: the download is capped, the unpacked tree must be too.
const MAX_UNPACKED_BYTES: u64 = 4 * 1024 * 1024 * 1024;
/// Written last, so a half-unpacked directory is never mistaken for a finished one.
const MARKER_FILE: &str = ".installed";
const INSTALL_DIR: &str = "acp-agents";

static HTTP_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder()
        .timeout(DOWNLOAD_TIMEOUT)
        // Registry URLs are HTTPS today; this also refuses an HTTPS -> HTTP
        // redirect, which no check on the URL string itself would catch.
        .https_only(true)
        .user_agent(concat!("vibe-kanban/", env!("CARGO_PKG_VERSION")))
        .build()
        .expect("failed to build ACP archive download client")
});

/// Download progress for an agent, for the settings UI.
#[derive(Debug, Clone, Default, Serialize)]
pub struct InstallProgress {
    pub downloaded: u64,
    pub total: Option<u64>,
    pub done: bool,
    pub error: Option<String>,
}

static PROGRESS: LazyLock<StdMutex<HashMap<String, InstallProgress>>> =
    LazyLock::new(|| StdMutex::new(HashMap::new()));

/// Serialises installs of the same agent, so a second spawn or install waits for
/// the first instead of downloading the archive again.
static INSTALL_LOCKS: LazyLock<StdMutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>> =
    LazyLock::new(|| StdMutex::new(HashMap::new()));

/// Latest known progress for `agent_id`, if an install was started this session.
pub fn progress(agent_id: &str) -> Option<InstallProgress> {
    PROGRESS.lock().ok()?.get(agent_id).cloned()
}

fn set_progress(agent_id: &str, value: InstallProgress) {
    if let Ok(mut map) = PROGRESS.lock() {
        map.insert(agent_id.to_string(), value);
    }
}

fn update_downloaded(agent_id: &str, downloaded: u64) {
    if let Ok(mut map) = PROGRESS.lock()
        && let Some(progress) = map.get_mut(agent_id)
    {
        progress.downloaded = downloaded;
    }
}

fn install_lock(agent_id: &str) -> Arc<tokio::sync::Mutex<()>> {
    let mut locks = INSTALL_LOCKS
        .lock()
        .expect("ACP install lock registry poisoned");
    locks
        .entry(agent_id.to_string())
        .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
        .clone()
}

#[derive(Debug, Error)]
pub enum InstallError {
    #[error("failed to download agent archive {url}: {source}")]
    Download {
        url: String,
        #[source]
        source: reqwest::Error,
    },
    #[error("agent archive {url} is larger than {MAX_ARCHIVE_BYTES} bytes")]
    TooLarge { url: String },
    #[error("refusing to download agent archive over insecure URL {url}")]
    InsecureUrl { url: String },
    #[error("agent archive {url} has no sha256 in the registry; refusing to install it unverified")]
    MissingChecksum { url: String },
    #[error("unpacked agent archive exceeds {MAX_UNPACKED_BYTES} bytes")]
    UnpackTooLarge,
    #[error(
        "{id} is not installed, or needs an update after a registry change; install or update it in Settings → Available agents"
    )]
    NotInstalled { id: String },
    #[error("checksum mismatch for {url}: expected {expected}, got {actual}")]
    Checksum {
        url: String,
        expected: String,
        actual: String,
    },
    #[error("archive entry escapes the install directory: {path}")]
    UnsafeEntry { path: String },
    #[error("agent archive does not contain the command '{cmd}'")]
    MissingCommand { cmd: String },
    #[error("failed to unpack agent archive: {0}")]
    Unpack(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// A ready-to-run binary and the arguments/env the registry gives it.
#[derive(Debug, Clone)]
pub struct InstalledAgent {
    pub command: PathBuf,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
}

/// Registry platform key for the machine this build runs on.
pub fn platform_key() -> Option<&'static str> {
    let musl = cfg!(all(target_os = "linux", target_env = "musl"));
    platform_key_for(std::env::consts::OS, std::env::consts::ARCH, musl)
}

fn platform_key_for(os: &str, arch: &str, musl: bool) -> Option<&'static str> {
    Some(match (os, arch) {
        ("windows", "x86_64") => "windows-x86_64",
        ("windows", "aarch64") => "windows-aarch64",
        ("macos", "aarch64") => "darwin-aarch64",
        ("macos", "x86_64") => "darwin-x86_64",
        ("linux", "x86_64") if musl => "linux-x86_64-musl",
        ("linux", "x86_64") => "linux-x86_64",
        ("linux", "aarch64") if musl => "linux-aarch64-musl",
        ("linux", "aarch64") => "linux-aarch64",
        _ => return None,
    })
}

/// Downloads and unpacks the agent if needed, then returns its command.
///
/// Only the settings install route calls this: spawning must not download,
/// so the spawn and discovery paths use [`installed`] instead.
///
/// Idempotent: a finished install (marker file + command present) is reused and
/// nothing is fetched.
pub async fn ensure_installed(
    agent_id: &str,
    version: &str,
    platform: &str,
    target: &BinaryTarget,
) -> Result<InstalledAgent, InstallError> {
    let lock = install_lock(agent_id);
    let _guard = lock.lock().await;

    let result = install_locked(agent_id, version, platform, target).await;

    let mut snapshot = progress(agent_id).unwrap_or_default();
    snapshot.done = true;
    match &result {
        Ok(_) => snapshot.error = None,
        Err(err) => snapshot.error = Some(err.to_string()),
    }
    set_progress(agent_id, snapshot);

    result
}

/// The command of an already-installed agent, without any network access.
///
/// Used by the spawn and discovery paths: a spawn must never download (up to
/// 1 GiB with a 10-minute timeout), so a missing install fails with a pointer
/// to Settings instead.
///
/// The version on disk wins over the registry's current one: a registry bump
/// must not break an installed agent. Any finished `{version}-{platform}`
/// install counts, newest first, so Settings can install a newer version and
/// the old one keeps running until then.
pub fn installed(
    agent_id: &str,
    platform: &str,
    target: &BinaryTarget,
) -> Result<InstalledAgent, InstallError> {
    let agent_root = workspace_utils::cache_dir()
        .join(INSTALL_DIR)
        .join(safe_component(agent_id));
    let suffix = format!("-{platform}");

    let mut candidates: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(&agent_root)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            if name.starts_with('.') || !name.ends_with(&suffix) {
                return None;
            }
            let dir = entry.path();
            if !dir.join(MARKER_FILE).is_file() {
                return None;
            }
            let modified = entry.metadata().ok()?.modified().ok()?;
            Some((modified, dir))
        })
        .collect();
    candidates.sort_by(|a, b| b.0.cmp(&a.0));

    candidates
        .into_iter()
        .find_map(|(_, dir)| installed_agent(target, &dir).ok())
        .ok_or_else(|| InstallError::NotInstalled {
            id: agent_id.to_string(),
        })
}

/// The finished install under `install_dir`, if the marker and the command are
/// both there. Broken installs are reported as absent so the caller can repair
/// them (downloads) or refuse (spawns).
fn install_from_disk(
    agent_id: &str,
    version: &str,
    platform: &str,
    target: &BinaryTarget,
) -> Result<InstalledAgent, InstallError> {
    let dir = install_dir(agent_id, version, platform);
    if dir.join(MARKER_FILE).is_file() && contained_command(&dir, &target.cmd).is_ok() {
        return installed_agent(target, &dir);
    }
    Err(InstallError::NotInstalled {
        id: agent_id.to_string(),
    })
}

async fn install_locked(
    agent_id: &str,
    version: &str,
    platform: &str,
    target: &BinaryTarget,
) -> Result<InstalledAgent, InstallError> {
    if let Ok(agent) = install_from_disk(agent_id, version, platform, target) {
        return Ok(agent);
    }

    let bytes = download(&target.archive, agent_id).await?;
    verify_checksum(&target.archive, &bytes, target.sha256.as_deref())?;

    let dir = install_dir(agent_id, version, platform);
    let parent = dir
        .parent()
        .ok_or_else(|| InstallError::Unpack(format!("no parent for {}", dir.display())))?;
    tokio::fs::create_dir_all(parent).await?;

    // Unpack next to the target and rename at the end, so a concurrent spawn
    // never sees a partial tree.
    let temp = parent.join(format!(
        ".install-{}-{}",
        safe_component(version),
        uuid::Uuid::new_v4()
    ));
    tokio::fs::create_dir_all(&temp).await?;

    let unpacked = {
        let temp = temp.clone();
        let target = target.clone();
        tokio::task::spawn_blocking(move || unpack(&bytes, &target, &temp))
            .await
            .map_err(|e| InstallError::Unpack(e.to_string()))?
    };
    if let Err(e) = unpacked {
        let _ = tokio::fs::remove_dir_all(&temp).await;
        return Err(e);
    }

    set_executable(&contained_command(&temp, &target.cmd)?)?;
    tokio::fs::write(temp.join(MARKER_FILE), b"ok").await?;

    match tokio::fs::rename(&temp, &dir).await {
        Ok(()) => prune_other_versions(&dir, platform).await,
        Err(_) => {
            // Another finished install may own the directory; a broken leftover
            // (no marker or no command) is cleared so this install replaces it.
            let usable =
                dir.join(MARKER_FILE).is_file() && contained_command(&dir, &target.cmd).is_ok();
            if !usable {
                let _ = tokio::fs::remove_dir_all(&dir).await;
                if let Err(e) = tokio::fs::rename(&temp, &dir).await {
                    let _ = tokio::fs::remove_dir_all(&temp).await;
                    return Err(e.into());
                }
                prune_other_versions(&dir, platform).await;
            } else {
                let _ = tokio::fs::remove_dir_all(&temp).await;
            }
        }
    }

    installed_agent(target, &dir)
}

/// Drops the other cached versions of this agent for the same platform: every
/// registry bump installs a new `{version}-{platform}` directory, and archives
/// are 100 MB+ each.
async fn prune_other_versions(current: &Path, platform: &str) {
    let Some(parent) = current.parent() else {
        return;
    };
    let suffix = format!("-{platform}");
    let Ok(mut entries) = tokio::fs::read_dir(parent).await else {
        return;
    };
    let mut stale = Vec::new();
    while let Ok(Some(entry)) = entries.next_entry().await {
        let Some(name) = entry.file_name().into_string().ok() else {
            continue;
        };
        // Skip in-flight downloads (`.install-*` temp dirs) and other platforms.
        if !name.starts_with('.') && name.ends_with(&suffix) && entry.path() != current {
            stale.push(entry.path());
        }
    }
    for path in stale {
        let _ = tokio::fs::remove_dir_all(&path).await;
    }
}

/// Quotes a path for [`crate::command::CommandBuilder`], which splits its base
/// command into words. Install paths live under the user's cache dir, whose
/// path may contain spaces.
pub fn command_line(path: &Path) -> String {
    let path = path.to_string_lossy();
    #[cfg(unix)]
    {
        // `shlex` re-splits the base, so quote with the same parser: a plain
        // surrounding `"` breaks on a cache path containing `'` or `\`.
        match shlex::try_quote(&path) {
            Ok(quoted) => quoted.into_owned(),
            Err(_) => path.into_owned(),
        }
    }
    #[cfg(not(unix))]
    {
        if path.chars().any(char::is_whitespace) {
            format!("\"{path}\"")
        } else {
            path.into_owned()
        }
    }
}

fn installed_agent(target: &BinaryTarget, dir: &Path) -> Result<InstalledAgent, InstallError> {
    Ok(InstalledAgent {
        command: contained_command(dir, &target.cmd)?,
        args: target.args.clone(),
        env: target.env.clone(),
    })
}

fn install_dir(agent_id: &str, version: &str, platform: &str) -> PathBuf {
    workspace_utils::cache_dir()
        .join(INSTALL_DIR)
        .join(safe_component(agent_id))
        .join(format!("{}-{}", safe_component(version), platform))
}

/// Registry ids are `[a-z0-9-]` and versions are semver, but the registry is
/// remote input: never let either escape the cache dir.
fn safe_component(value: &str) -> String {
    let cleaned: String = value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.is_empty() || cleaned.chars().all(|c| c == '.') {
        "_".to_string()
    } else {
        cleaned
    }
}

fn command_path(dir: &Path, cmd: &str) -> Result<PathBuf, InstallError> {
    let relative = cmd
        .strip_prefix("./")
        .or_else(|| cmd.strip_prefix(".\\"))
        .unwrap_or(cmd);
    let relative = Path::new(relative);

    // The path is later re-split by a shell-like parser (`shlex` on Unix,
    // `winsplit` on Windows), so quotes would smuggle components past the
    // check below: `x/".."/".."/bin/sh` passes `Component::Normal` and then
    // unquotes to `x/../../bin/sh`.
    if cmd.contains('"') || cmd.contains('\'') || cmd.chars().any(char::is_whitespace) {
        return Err(InstallError::UnsafeEntry {
            path: cmd.to_string(),
        });
    }
    #[cfg(not(windows))]
    if cmd.contains('\\') {
        return Err(InstallError::UnsafeEntry {
            path: cmd.to_string(),
        });
    }

    let escapes = relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)));
    if escapes {
        return Err(InstallError::UnsafeEntry {
            path: cmd.to_string(),
        });
    }

    Ok(dir.join(relative))
}

/// The command path resolved through symlinks and required to stay inside the
/// install directory. Archives can ship a symlink pointing anywhere, and both
/// the chmod and the actual run follow links, so a link to e.g.
/// `~/.ssh/id_rsa` must never be made executable.
fn contained_command(dir: &Path, cmd: &str) -> Result<PathBuf, InstallError> {
    let path = command_path(dir, cmd)?;
    let resolved = path
        .canonicalize()
        .map_err(|_| InstallError::MissingCommand {
            cmd: cmd.to_string(),
        })?;
    let root = dir
        .canonicalize()
        .map_err(|_| InstallError::MissingCommand {
            cmd: cmd.to_string(),
        })?;
    if !resolved.starts_with(&root) {
        return Err(InstallError::UnsafeEntry {
            path: cmd.to_string(),
        });
    }
    // Run the original, non-canonicalized path: on Windows `canonicalize`
    // returns `\\?\C:\…` verbatim paths, which break `.cmd` launchers
    // (Cursor). The resolved path is only used for the containment check.
    Ok(path)
}

async fn download(url: &str, agent_id: &str) -> Result<Vec<u8>, InstallError> {
    // `https_only` on the client also refuses HTTPS -> HTTP redirects; this
    // check catches the plain-HTTP URL itself with an error that names it.
    if !url.starts_with("https://") {
        return Err(InstallError::InsecureUrl {
            url: url.to_string(),
        });
    }
    let response = HTTP_CLIENT
        .get(url)
        .send()
        .await
        .map_err(|source| InstallError::Download {
            url: url.to_string(),
            source,
        })?;
    let response = response
        .error_for_status()
        .map_err(|source| InstallError::Download {
            url: url.to_string(),
            source,
        })?;

    let total = response.content_length();
    if let Some(length) = total
        && length > MAX_ARCHIVE_BYTES
    {
        return Err(InstallError::TooLarge {
            url: url.to_string(),
        });
    }
    set_progress(
        agent_id,
        InstallProgress {
            total,
            ..Default::default()
        },
    );

    let mut bytes = Vec::with_capacity(total.unwrap_or_default().min(MAX_ARCHIVE_BYTES) as usize);
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|source| InstallError::Download {
            url: url.to_string(),
            source,
        })?;
        bytes.extend_from_slice(&chunk);

        if bytes.len() as u64 > MAX_ARCHIVE_BYTES {
            return Err(InstallError::TooLarge {
                url: url.to_string(),
            });
        }
        update_downloaded(agent_id, bytes.len() as u64);
    }

    Ok(bytes)
}

fn verify_checksum(url: &str, bytes: &[u8], expected: Option<&str>) -> Result<(), InstallError> {
    let Some(expected) = expected else {
        // The registry is remote input: without a checksum the archive is
        // whatever the host serves at that moment, so refuse to install it.
        return Err(InstallError::MissingChecksum {
            url: url.to_string(),
        });
    };

    let actual = format!("{:x}", Sha256::digest(bytes));
    if actual.eq_ignore_ascii_case(expected.trim()) {
        Ok(())
    } else {
        Err(InstallError::Checksum {
            url: url.to_string(),
            expected: expected.trim().to_string(),
            actual,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArchiveKind {
    Zip,
    TarGz,
    TarBz2,
    Raw,
}

fn archive_kind(url: &str) -> ArchiveKind {
    let path = url
        .split(['?', '#'])
        .next()
        .unwrap_or(url)
        .to_ascii_lowercase();

    if path.ends_with(".zip") {
        ArchiveKind::Zip
    } else if path.ends_with(".tar.gz") || path.ends_with(".tgz") {
        ArchiveKind::TarGz
    } else if path.ends_with(".tar.bz2") || path.ends_with(".tbz2") {
        ArchiveKind::TarBz2
    } else {
        ArchiveKind::Raw
    }
}

fn unpack(bytes: &[u8], target: &BinaryTarget, dest: &Path) -> Result<(), InstallError> {
    unpack_capped(bytes, target, dest, MAX_UNPACKED_BYTES)
}

/// `cap` bounds the bytes written to disk: only the download is size-limited,
/// so without this a zip bomb would fill the disk. Test seam for the cap.
fn unpack_capped(
    bytes: &[u8],
    target: &BinaryTarget,
    dest: &Path,
    cap: u64,
) -> Result<(), InstallError> {
    match archive_kind(&target.archive) {
        ArchiveKind::Zip => unpack_zip(bytes, dest, cap),
        ArchiveKind::TarGz => {
            unpack_tar(flate2::read::GzDecoder::new(Cursor::new(bytes)), dest, cap)
        }
        ArchiveKind::TarBz2 => {
            unpack_tar(bzip2::read::BzDecoder::new(Cursor::new(bytes)), dest, cap)
        }
        ArchiveKind::Raw => {
            if bytes.len() as u64 > cap {
                return Err(InstallError::UnpackTooLarge);
            }
            let path = command_path(dest, &target.cmd)?;
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, bytes)?;
            Ok(())
        }
    }
}

/// The entry's path under `dest`, after checking that no component of it is
/// a symlink we (or a previous entry) created. `enclosed_name` alone is not
/// enough: an earlier entry can plant `lib -> ~/.ssh` and a later
/// `lib/authorized_keys` would then be written *through* the link, outside
/// `dest`, at install time. Refusing to traverse links at all also keeps the
/// permission restore from touching anything outside `dest`.
fn safe_entry_path(dest: &Path, relative: &Path) -> Result<PathBuf, InstallError> {
    let mut prefix = dest.to_path_buf();
    for component in relative.components() {
        prefix.push(component);
        if let Ok(meta) = std::fs::symlink_metadata(&prefix)
            && meta.file_type().is_symlink()
        {
            return Err(InstallError::UnsafeEntry {
                path: relative.display().to_string(),
            });
        }
    }
    Ok(dest.join(relative))
}

/// A symlink target is only allowed when it is relative and stays inside
/// `dest` after resolving `.`/`..` lexically. Together with
/// [`safe_entry_path`] (nothing is ever written through a link) that makes
/// every link we create resolve inside `dest` at run time too, including
/// link-to-link chains: each link's lexical target stays inside, so a chain
/// can never leave.
fn symlink_target_stays_inside(dest: &Path, link_dir: &Path, target: &str) -> bool {
    let target = target.trim_end_matches('\0');
    let relative = Path::new(target);
    if relative.is_absolute() {
        return false;
    }
    let mut resolved = link_dir.to_path_buf();
    for component in relative.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::Normal(part) => resolved.push(part),
            // `..` is allowed while it stays at or inside `dest` (e.g.
            // `bin/tool -> ../lib/tool`); anything that would climb above
            // `dest` is rejected.
            std::path::Component::ParentDir => {
                resolved.pop();
                if !resolved.starts_with(dest) {
                    return false;
                }
            }
            // Absolute targets (and Windows prefixes) resolve outside by
            // definition.
            _ => return false,
        }
    }
    resolved.starts_with(dest)
}

fn unpack_zip(bytes: &[u8], dest: &Path, cap: u64) -> Result<(), InstallError> {
    use std::io::Write;

    let mut archive = zip::ZipArchive::new(Cursor::new(bytes))
        .map_err(|e| InstallError::Unpack(e.to_string()))?;

    let mut written: u64 = 0;
    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .map_err(|e| InstallError::Unpack(e.to_string()))?;

        let Some(relative) = entry.enclosed_name() else {
            return Err(InstallError::UnsafeEntry {
                path: entry.name().to_string(),
            });
        };
        let out = safe_entry_path(dest, &relative)?;

        if entry.is_dir() {
            std::fs::create_dir_all(&out)?;
            apply_unix_mode(&out, entry.unix_mode())?;
            continue;
        }

        // Zip stores neither Unix modes nor symlinks unless they are written
        // explicitly; the macOS `.app` zips (junie) need both, so restore them
        // instead of leaving everything 0o644 regular files.
        if entry.is_symlink() {
            // `take` bounds the read: declared sizes in the central directory
            // cannot be trusted to feed the cap accounting.
            let budget = cap - written + 1;
            let mut limited = (&mut entry).take(budget);
            let mut link_target = String::new();
            limited
                .read_to_string(&mut link_target)
                .map_err(|e| InstallError::Unpack(e.to_string()))?;
            written += budget - limited.limit();
            if written > cap {
                return Err(InstallError::UnpackTooLarge);
            }
            #[cfg(unix)]
            {
                if !symlink_target_stays_inside(dest, out.parent().unwrap_or(dest), &link_target) {
                    return Err(InstallError::UnsafeEntry {
                        path: relative.display().to_string(),
                    });
                }
                if let Some(parent) = out.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::os::unix::fs::symlink(link_target.trim_end_matches('\0'), &out)?;
            }
            #[cfg(not(unix))]
            {
                let _ = link_target; // no trustworthy symlink flag on Windows; skip
            }
            continue;
        }

        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::File::create(&out)?;
        // Bound the actual copy, not the declared size: a crafted archive can
        // understate it.
        let mut limited = (&mut entry).take(cap - written + 1);
        let copied = std::io::copy(&mut limited, &mut file)?;
        written += copied;
        if written > cap {
            return Err(InstallError::UnpackTooLarge);
        }
        file.flush()?;
        apply_unix_mode(&out, entry.unix_mode())?;
    }

    Ok(())
}

#[cfg(unix)]
fn apply_unix_mode(path: &Path, mode: Option<u32>) -> Result<(), InstallError> {
    use std::os::unix::fs::PermissionsExt;

    if let Some(mode) = mode {
        // 0o777, not 0o7777: no setuid/setgid/sticky from an archive.
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode & 0o777))?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn apply_unix_mode(_path: &Path, _mode: Option<u32>) -> Result<(), InstallError> {
    Ok(())
}

fn unpack_tar<R: Read>(reader: R, dest: &Path, cap: u64) -> Result<(), InstallError> {
    std::fs::create_dir_all(dest)?;
    let mut written: u64 = 0;
    let mut archive = tar::Archive::new(reader);
    let mut entries = archive
        .entries()
        .map_err(|e| InstallError::Unpack(e.to_string()))?;
    // Per-entry `unpack_in` keeps tar's own protections: modes, symlinks and
    // refusal to escape `dest`, same as `Archive::unpack`. A `false` return
    // means the entry was skipped as unsafe, which `Archive::unpack` would
    // have turned into an error too.
    while let Some(mut entry) = entries
        .next()
        .transpose()
        .map_err(|e| InstallError::Unpack(e.to_string()))?
    {
        written += entry.header().size()?;
        if written > cap {
            return Err(InstallError::UnpackTooLarge);
        }
        let unpacked = entry
            .unpack_in(dest)
            .map_err(|e| InstallError::Unpack(e.to_string()))?;
        if !unpacked {
            let path = entry
                .path()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| "<unreadable>".to_string());
            return Err(InstallError::UnsafeEntry { path });
        }
    }
    Ok(())
}

#[cfg(unix)]
fn set_executable(path: &Path) -> Result<(), InstallError> {
    use std::os::unix::fs::PermissionsExt;

    let mut permissions = std::fs::metadata(path)
        .map_err(|_| InstallError::MissingCommand {
            cmd: path.display().to_string(),
        })?
        .permissions();
    permissions.set_mode(permissions.mode() | 0o755);
    std::fs::set_permissions(path, permissions)?;
    Ok(())
}

#[cfg(not(unix))]
fn set_executable(path: &Path) -> Result<(), InstallError> {
    if !path.is_file() {
        return Err(InstallError::MissingCommand {
            cmd: path.display().to_string(),
        });
    }
    Ok(())
}

/// Deletes every cached archive of `agent_id`, all versions and platforms.
/// Best-effort by design: uninstall must succeed even when the files are
/// already gone.
pub fn remove_agent_files(agent_id: &str) -> std::io::Result<()> {
    let dir = workspace_utils::cache_dir()
        .join(INSTALL_DIR)
        .join(safe_component(agent_id));
    match std::fs::remove_dir_all(&dir) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    fn target(archive: &str) -> BinaryTarget {
        BinaryTarget {
            archive: archive.to_string(),
            sha256: None,
            cmd: "./bin/agent".to_string(),
            args: vec!["acp".to_string()],
            env: HashMap::from([("AGENT_MODE".to_string(), "1".to_string())]),
        }
    }

    #[test]
    fn platform_keys_match_the_registry() {
        assert_eq!(
            platform_key_for("windows", "x86_64", false),
            Some("windows-x86_64")
        );
        assert_eq!(
            platform_key_for("windows", "aarch64", false),
            Some("windows-aarch64")
        );
        assert_eq!(
            platform_key_for("macos", "aarch64", false),
            Some("darwin-aarch64")
        );
        assert_eq!(
            platform_key_for("linux", "x86_64", false),
            Some("linux-x86_64")
        );
        assert_eq!(
            platform_key_for("linux", "aarch64", true),
            Some("linux-aarch64-musl")
        );
        assert_eq!(platform_key_for("freebsd", "x86_64", false), None);
    }

    #[test]
    fn archive_kind_ignores_query_strings() {
        assert_eq!(archive_kind("https://x/a.zip"), ArchiveKind::Zip);
        assert_eq!(archive_kind("https://x/a.zip?token=1"), ArchiveKind::Zip);
        assert_eq!(archive_kind("https://x/a.tar.gz"), ArchiveKind::TarGz);
        assert_eq!(archive_kind("https://x/a.tgz#frag"), ArchiveKind::TarGz);
        assert_eq!(archive_kind("https://x/a.tar.bz2"), ArchiveKind::TarBz2);
        assert_eq!(archive_kind("https://x/a.tbz2"), ArchiveKind::TarBz2);
        assert_eq!(archive_kind("https://x/opencode.exe"), ArchiveKind::Raw);
    }

    #[test]
    fn command_path_strips_dot_prefixes() {
        let dir = Path::new("/tmp/agent");
        assert_eq!(
            command_path(dir, "./opencode.exe").unwrap(),
            dir.join("opencode.exe")
        );
        assert_eq!(command_path(dir, "goose").unwrap(), dir.join("goose"));
        #[cfg(windows)]
        assert_eq!(
            command_path(dir, ".\\bin\\devin.exe").unwrap(),
            dir.join("bin/devin.exe")
        );
    }

    #[test]
    fn command_path_rejects_escapes() {
        let dir = Path::new("/tmp/agent");
        for cmd in [
            "../evil",
            "/usr/bin/evil",
            "a/../../evil",
            // Quotes pass the `Component::Normal` check but unquote to `..`
            // when the base command is re-split by `shlex`/`winsplit`.
            "x/\"..\"/\"..\"/bin/sh",
            "a b",
            "a'b",
        ] {
            assert!(
                matches!(
                    command_path(dir, cmd),
                    Err(InstallError::UnsafeEntry { .. })
                ),
                "{cmd} must be rejected"
            );
        }
        #[cfg(windows)]
        assert!(matches!(
            command_path(dir, "C:\\evil.exe"),
            Err(InstallError::UnsafeEntry { .. })
        ));
    }

    #[test]
    #[cfg(not(windows))]
    fn command_path_rejects_backslashes_on_unix() {
        assert!(matches!(
            command_path(Path::new("/tmp/agent"), "bin\\agent"),
            Err(InstallError::UnsafeEntry { .. })
        ));
    }

    #[test]
    fn contained_command_rejects_symlinks_out_of_the_install_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("bin")).unwrap();
        std::fs::write(dir.path().join("bin/agent"), b"agent").unwrap();
        // In-tree link resolves inside the dir: allowed.
        assert!(contained_command(dir.path(), "bin/agent").is_ok());

        #[cfg(unix)]
        {
            let outside = tempfile::tempdir().unwrap();
            std::os::unix::fs::symlink(outside.path(), dir.path().join("bin/link")).unwrap();
            assert!(matches!(
                contained_command(dir.path(), "bin/link"),
                Err(InstallError::UnsafeEntry { .. })
            ));
        }
    }

    #[test]
    fn safe_component_never_escapes() {
        assert_eq!(safe_component("opencode-v2"), "opencode-v2");
        assert_eq!(safe_component("../.."), ".._..");
        assert_eq!(safe_component(".."), "_");
        assert_eq!(safe_component("a/b"), "a_b");
    }

    #[test]
    fn checksum_is_verified() {
        let bytes = b"hello";
        let digest = format!("{:x}", Sha256::digest(bytes));
        assert!(verify_checksum("https://x/a.zip", bytes, Some(&digest)).is_ok());
        assert!(matches!(
            verify_checksum("https://x/a.zip", bytes, Some(&"0".repeat(64))),
            Err(InstallError::Checksum { .. })
        ));
    }

    #[test]
    fn missing_checksum_is_rejected() {
        assert!(matches!(
            verify_checksum("https://x/a.zip", b"hello", None),
            Err(InstallError::MissingChecksum { .. })
        ));
    }

    #[tokio::test]
    async fn plain_http_downloads_are_refused() {
        // Fails on the scheme guard before any network access happens; a
        // redirect to http:// is refused by `https_only` on the client.
        for url in ["http://x/a.zip", "ftp://x/a.zip", "https:x/a.zip"] {
            assert!(
                matches!(
                    download(url, "test").await,
                    Err(InstallError::InsecureUrl { .. })
                ),
                "{url} must be refused"
            );
        }
    }

    #[test]
    fn zip_archive_is_unpacked() {
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        writer.start_file("bin/agent", options).unwrap();
        writer.write_all(b"agent").unwrap();
        let bytes = writer.finish().unwrap().into_inner();

        let dest = tempfile::tempdir().unwrap();
        unpack(&bytes, &target("https://x/a.zip"), dest.path()).unwrap();
        assert_eq!(
            std::fs::read(dest.path().join("bin/agent")).unwrap(),
            b"agent"
        );
    }

    #[test]
    fn zip_archive_cannot_write_outside_destination() {
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        writer
            .start_file("../evil.txt", zip::write::SimpleFileOptions::default())
            .unwrap();
        writer.write_all(b"evil").unwrap();
        let bytes = writer.finish().unwrap().into_inner();

        let dest = tempfile::tempdir().unwrap();
        assert!(matches!(
            unpack(&bytes, &target("https://x/a.zip"), dest.path()),
            Err(InstallError::UnsafeEntry { .. })
        ));
        assert!(!dest.path().parent().unwrap().join("evil.txt").exists());
    }

    #[test]
    fn tar_gz_archive_is_unpacked() {
        let encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        let mut builder = tar::Builder::new(encoder);

        let mut header = tar::Header::new_gnu();
        header.set_size(5);
        header.set_mode(0o755);
        header.set_cksum();
        builder
            .append_data(&mut header, "bin/agent", &b"agent"[..])
            .unwrap();

        let bytes = builder.into_inner().unwrap().finish().unwrap();
        let dest = tempfile::tempdir().unwrap();
        unpack(&bytes, &target("https://x/a.tar.gz"), dest.path()).unwrap();
        assert!(dest.path().join("bin/agent").is_file());
    }

    #[test]
    fn raw_binary_is_written_to_the_command_path() {
        let dest = tempfile::tempdir().unwrap();
        unpack(b"raw", &target("https://x/agent.exe"), dest.path()).unwrap();
        assert_eq!(
            std::fs::read(dest.path().join("bin/agent")).unwrap(),
            b"raw"
        );
    }

    #[test]
    fn command_line_round_trips_through_the_command_splitter() {
        // The base command is re-split by `shlex`/`winsplit`, so whatever we
        // quote must come back out as a single word (spaces, quotes, and on
        // Unix backslashes are the interesting cache-dir characters).
        for path in ["/opt/agent", "/opt/my agent", "/o'pt/agent", "/opt/a\\b"] {
            let quoted = command_line(Path::new(path));
            let built = crate::command::CommandBuilder::new(quoted)
                .build_initial()
                .unwrap_or_else(|e| panic!("{path}: {e}"));
            assert_eq!(built.program, path);
        }
    }

    #[test]
    fn zip_unpack_keeps_symlinks_and_modes() {
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .unix_permissions(0o755);
        writer.start_file("bin/agent", options).unwrap();
        writer.write_all(b"agent").unwrap();
        writer.add_symlink("bin/latest", "agent", options).unwrap();
        let bytes = writer.finish().unwrap().into_inner();

        let dest = tempfile::tempdir().unwrap();
        unpack(&bytes, &target("https://x/a.zip"), dest.path()).unwrap();

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(dest.path().join("bin/agent"))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o111, 0o111, "executable bit must survive");
            assert!(
                std::fs::symlink_metadata(dest.path().join("bin/latest"))
                    .unwrap()
                    .file_type()
                    .is_symlink()
            );
        }
    }

    #[test]
    fn unpack_is_capped() {
        // Raw path over the cap.
        let dest = tempfile::tempdir().unwrap();
        assert!(matches!(
            unpack_capped(b"raw", &target("https://x/agent.exe"), dest.path(), 2),
            Err(InstallError::UnpackTooLarge)
        ));

        // Zip path over the cap: two 3-byte files against a 4-byte cap.
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        let options = zip::write::SimpleFileOptions::default();
        for name in ["a", "b"] {
            writer.start_file(name, options).unwrap();
            writer.write_all(b"abc").unwrap();
        }
        let bytes = writer.finish().unwrap().into_inner();
        let dest = tempfile::tempdir().unwrap();
        assert!(matches!(
            unpack_capped(&bytes, &target("https://x/a.zip"), dest.path(), 4),
            Err(InstallError::UnpackTooLarge)
        ));
    }

    /// Rewrites the fixed header of the first central-directory entry, so a
    /// test can craft archives the writer API refuses to produce (lying
    /// sizes, setuid modes).
    fn patch_first_cd_entry(bytes: &mut Vec<u8>, write: impl FnOnce(&mut [u8])) {
        let eocd = bytes
            .windows(4)
            .rposition(|window| window == b"PK\x05\x06")
            .expect("end of central directory");
        let cd_offset =
            u32::from_le_bytes(bytes[eocd + 16..eocd + 20].try_into().unwrap()) as usize;
        assert_eq!(&bytes[cd_offset..cd_offset + 4], b"PK\x01\x02");
        write(&mut bytes[cd_offset..cd_offset + 46]);
    }

    #[test]
    fn zip_understated_sizes_are_caught_by_the_copy_limit() {
        // 5 actual bytes, declared as 1: the declared-size accounting would
        // pass a 4-byte cap, the bounded copy must not.
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        writer
            .start_file(
                "big",
                zip::write::SimpleFileOptions::default()
                    .compression_method(zip::CompressionMethod::Deflated),
            )
            .unwrap();
        writer.write_all(b"xxxxx").unwrap();
        let mut bytes = writer.finish().unwrap().into_inner();
        patch_first_cd_entry(&mut bytes, |entry| {
            // compressed size (offset 20) stays true so the reader streams
            // everything; only the uncompressed size (offset 24) lies.
            entry[24..28].copy_from_slice(&1u32.to_le_bytes());
        });

        let dest = tempfile::tempdir().unwrap();
        assert!(matches!(
            unpack_capped(&bytes, &target("https://x/a.zip"), dest.path(), 4),
            Err(InstallError::UnpackTooLarge)
        ));
    }

    #[test]
    #[cfg(unix)]
    fn zip_setuid_bits_are_dropped() {
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        writer
            .start_file("bin/agent", zip::write::SimpleFileOptions::default())
            .unwrap();
        writer.write_all(b"agent").unwrap();
        let mut bytes = writer.finish().unwrap().into_inner();
        patch_first_cd_entry(&mut bytes, |entry| {
            // external attributes = S_IFREG | setuid | rwxr-xr-x in the high word
            let mode: u32 = 0o104_755;
            entry[38..42].copy_from_slice(&(mode << 16).to_le_bytes());
        });

        let dest = tempfile::tempdir().unwrap();
        unpack(&bytes, &target("https://x/a.zip"), dest.path()).unwrap();
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(dest.path().join("bin/agent"))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o7000, 0, "setuid/setgid/sticky must not survive");
        assert_eq!(mode & 0o777, 0o755);
    }

    #[test]
    #[cfg(unix)]
    fn zip_symlinks_cannot_escape_or_be_written_through() {
        let options = zip::write::SimpleFileOptions::default();

        // A target that climbs out of the install dir is refused.
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        writer.add_symlink("up", "..", options).unwrap();
        let bytes = writer.finish().unwrap().into_inner();
        let dest = tempfile::tempdir().unwrap();
        assert!(matches!(
            unpack(&bytes, &target("https://x/a.zip"), dest.path()),
            Err(InstallError::UnsafeEntry { .. })
        ));

        // So is an absolute target.
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        writer.add_symlink("abs", "/etc/passwd", options).unwrap();
        let bytes = writer.finish().unwrap().into_inner();
        let dest = tempfile::tempdir().unwrap();
        assert!(matches!(
            unpack(&bytes, &target("https://x/a.zip"), dest.path()),
            Err(InstallError::UnsafeEntry { .. })
        ));

        // A legitimate in-tree link is created, but a later entry writing
        // *through* it is refused: without that check, `lib -> ~/.ssh` plus
        // `lib/authorized_keys` writes outside the install dir at install
        // time.
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        writer.add_symlink("lib", "inner", options).unwrap();
        writer.start_file("lib/authorized_keys", options).unwrap();
        writer.write_all(b"evil").unwrap();
        let bytes = writer.finish().unwrap().into_inner();
        let dest = tempfile::tempdir().unwrap();
        assert!(matches!(
            unpack(&bytes, &target("https://x/a.zip"), dest.path()),
            Err(InstallError::UnsafeEntry { .. })
        ));
        assert!(
            std::fs::symlink_metadata(dest.path().join("lib"))
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert!(!dest.path().join("inner").exists());
    }
}
