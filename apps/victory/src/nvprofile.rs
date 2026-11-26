//! `victory nvprofile`: an NVIDIA driver profile for `victory.exe` that
//! raises the driver's reserve for the machine code of pipelines.
//!
//! NVIDIA's D3D12 driver keeps 512 MB of address space for the machine code
//! of the pipelines in use and removes the device (`0x887a0005`) when a
//! submission needs more. The title translates to tens of thousands of
//! pipelines, so the reserve runs out within seconds of entering the world.
//! Driver setting `0x001ABC0F` in an application profile raises it; the
//! driver's own profiles for big titles use 1 or 2 GiB (`0x80000000` is read
//! there, so the value is taken to be bytes). The idea and the setting are
//! from XCL's `xcl nvprofile` (ivall/xcl#2590); this is written anew against
//! the public NVAPI driver-settings (DRS) interface.
//!
//! Nothing here runs unless the user runs the command. It writes one profile
//! of its own, [`PROFILE`], holding only that setting and the executable, so
//! `remove` deletes exactly what `install` made. Writing driver settings
//! needs an elevated prompt once; the driver then matches the profile by the
//! executable's name in every later run, elevated or not.

use std::ffi::c_void;

use anyhow::{Context, Result, bail};
use clap::Subcommand;

/// The driver's reserve for pipeline machine code, in bytes.
const SETTING: u32 = 0x001A_BC0F;
/// The profile `install` makes and `remove` deletes.
const PROFILE: &str = "VIctory shader code reserve";
/// The driver's reserve without a profile.
const DEFAULT_MB: u32 = 512;
/// `NvAPI_UnicodeString`: 2048 UTF-16 units.
const UNICODE: usize = 2048;

#[derive(Subcommand, Clone, Copy)]
pub(crate) enum Action {
    /// Show the reserve the driver gives the executable (the default).
    Status,
    /// Put the executable in a profile of its own with a larger reserve
    /// (needs an administrator prompt once).
    Install {
        /// The reserve in MB.
        #[arg(long, default_value_t = 1024, value_parser = clap::value_parser!(u32).range(512..=2048))]
        mb: u32,
    },
    /// Delete the profile `install` made.
    Remove,
}

pub(crate) fn run(action: Option<Action>, exe: Option<String>) -> Result<()> {
    let exe = match exe {
        Some(exe) => exe,
        None => std::env::current_exe()?
            .file_name()
            .context("no file name for this executable")?
            .to_string_lossy()
            .into_owned(),
    };
    let api = Api::load()?;
    let session = Session::open(&api)?;
    match action.unwrap_or(Action::Status) {
        Action::Status => status(&session, &exe),
        Action::Install { mb } => install(&session, &exe, mb),
        Action::Remove => remove(&session),
    }
}

fn status(s: &Session, exe: &str) -> Result<()> {
    let Some((profile, ours)) = s.profile_of(exe)? else {
        println!("{exe}: in no driver profile; the driver's default reserve, {DEFAULT_MB} MB");
        println!("`victory nvprofile install` (from an administrator prompt) raises it");
        return Ok(());
    };
    let whose = if ours {
        format!("the profile \"{PROFILE}\"")
    } else {
        let (name, predefined) = s.profile_name(profile)?;
        format!("the driver profile \"{name}\"{}", if predefined { " (one the driver ships)" } else { "" })
    };
    match s.get(profile)? {
        Some(bytes) => println!("{exe}: in {whose}; reserve {} MB ({bytes:#x})", bytes >> 20),
        None => println!("{exe}: in {whose}, which does not set it; the default, {DEFAULT_MB} MB"),
    }
    Ok(())
}

fn install(s: &Session, exe: &str, mb: u32) -> Result<()> {
    let ours = match s.find_profile(PROFILE)? {
        Some(p) => p,
        None => s.create_profile(PROFILE)?,
    };
    match s.profile_of(exe)? {
        Some((p, _)) if p == ours => {}
        Some((p, _)) => bail!(
            "{exe} is already in the driver profile \"{}\"; set 0x{SETTING:08X} there (NVIDIA Profile Inspector) \
             or pass another name with --exe",
            s.profile_name(p)?.0
        ),
        None => s.add_application(ours, exe)?,
    }
    s.set(ours, reserve_bytes(mb))?;
    s.save()?;
    println!("{exe}: in the profile \"{PROFILE}\"; reserve {mb} MB from the next start");
    Ok(())
}

fn remove(s: &Session) -> Result<()> {
    let Some(ours) = s.find_profile(PROFILE)? else {
        println!("no profile \"{PROFILE}\": nothing to remove");
        return Ok(());
    };
    s.delete_profile(ours)?;
    s.save()?;
    println!("the profile \"{PROFILE}\" is deleted; the driver's default reserve, {DEFAULT_MB} MB, again");
    Ok(())
}

/// The setting's value for a reserve of `mb` MB (512 to 2048: it fits 32 bits).
fn reserve_bytes(mb: u32) -> u32 {
    assert!((512..=2048).contains(&mb));
    mb << 20
}

// ---------------------------------------------------------------- NVAPI

/// `NvAPI_Status` values this file tells apart.
const OK: i32 = 0;
const PROFILE_NOT_FOUND: i32 = -163;
const EXECUTABLE_NOT_FOUND: i32 = -166;
const SETTING_NOT_FOUND: i32 = -160;
const INVALID_USER_PRIVILEGE: i32 = -137;

/// `NVDRS_PROFILE_V1`.
#[repr(C)]
struct ProfileV1 {
    version: u32,
    name: [u16; UNICODE],
    gpu_support: u32,
    is_predefined: u32,
    apps: u32,
    settings: u32,
}

/// `NVDRS_APPLICATION_V1`.
#[repr(C)]
struct ApplicationV1 {
    version: u32,
    is_predefined: u32,
    name: [u16; UNICODE],
    friendly_name: [u16; UNICODE],
    launcher: [u16; UNICODE],
}

/// `NVDRS_SETTING_V1`; the two values are unions of a dword, a binary value
/// (a length and 4096 bytes) and a string, so 4100 bytes each.
#[repr(C)]
struct SettingV1 {
    version: u32,
    name: [u16; UNICODE],
    id: u32,
    kind: u32,
    location: u32,
    is_current_predefined: u32,
    is_predefined_valid: u32,
    predefined: [u32; 1025],
    current: [u32; 1025],
}

/// `MAKE_NVAPI_VERSION(type, 1)`.
const fn version_1<T>() -> u32 {
    size_of::<T>() as u32 | 1 << 16
}

/// A zeroed NVAPI structure (all integers, so zero is a valid value) with its
/// version set: on the heap, as each is over 4 KiB.
fn blank<T>(version: impl FnOnce(&mut T) -> &mut u32) -> Box<T> {
    // SAFETY: only used for the `#[repr(C)]` integer structs above.
    let mut value: Box<T> = Box::new(unsafe { std::mem::zeroed() });
    let v = version_1::<T>();
    *version(&mut value) = v;
    value
}

fn unicode(text: &str) -> Box<[u16; UNICODE]> {
    let mut out = Box::new([0u16; UNICODE]);
    for (slot, unit) in out.iter_mut().take(UNICODE - 1).zip(text.encode_utf16()) {
        *slot = unit;
    }
    out
}

type Handle = *mut c_void;
type Status = i32;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryW(name: *const u16) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
}

/// The DRS entry points, from `nvapi_QueryInterface` by their public ids.
struct Api {
    create_session: unsafe extern "C" fn(*mut Handle) -> Status,
    destroy_session: unsafe extern "C" fn(Handle) -> Status,
    load_settings: unsafe extern "C" fn(Handle) -> Status,
    save_settings: unsafe extern "C" fn(Handle) -> Status,
    find_profile_by_name: unsafe extern "C" fn(Handle, *const u16, *mut Handle) -> Status,
    find_application_by_name: unsafe extern "C" fn(Handle, *const u16, *mut Handle, *mut ApplicationV1) -> Status,
    create_profile: unsafe extern "C" fn(Handle, *mut ProfileV1, *mut Handle) -> Status,
    delete_profile: unsafe extern "C" fn(Handle, Handle) -> Status,
    create_application: unsafe extern "C" fn(Handle, Handle, *mut ApplicationV1) -> Status,
    get_setting: unsafe extern "C" fn(Handle, Handle, u32, *mut SettingV1) -> Status,
    set_setting: unsafe extern "C" fn(Handle, Handle, *mut SettingV1) -> Status,
    get_profile_info: unsafe extern "C" fn(Handle, Handle, *mut ProfileV1) -> Status,
}

impl Api {
    // (each field's type is the annotation)
    #[allow(clippy::missing_transmute_annotations)]
    fn load() -> Result<Self> {
        let name: Vec<u16> = "nvapi64.dll\0".encode_utf16().collect();
        // SAFETY: a NUL-terminated name; the library stays loaded for the process.
        let module = unsafe { LoadLibraryW(name.as_ptr()) };
        if module.is_null() {
            bail!("nvapi64.dll is not there: no NVIDIA driver on this PC");
        }
        // SAFETY: a NUL-terminated export name of a loaded module.
        let query = unsafe { GetProcAddress(module, c"nvapi_QueryInterface".as_ptr().cast()) };
        if query.is_null() {
            bail!("nvapi64.dll has no nvapi_QueryInterface");
        }
        // SAFETY: the export's documented signature.
        let query: unsafe extern "C" fn(u32) -> *mut c_void = unsafe { std::mem::transmute(query) };
        let get = |id: u32, what: &str| -> Result<*mut c_void> {
            // SAFETY: asking for an entry point by id.
            let f = unsafe { query(id) };
            anyhow::ensure!(!f.is_null(), "this NVIDIA driver has no {what}");
            Ok(f)
        };
        let initialize: unsafe extern "C" fn() -> Status =
            // SAFETY: NvAPI_Initialize takes nothing and returns a status.
            unsafe { std::mem::transmute(get(0x0150_E828, "NvAPI_Initialize")?) };
        // SAFETY: as above.
        check(unsafe { initialize() }, "NvAPI_Initialize")?;
        // SAFETY (every transmute below): each id is the public one of the
        // entry point whose C signature the field's type spells.
        unsafe {
            Ok(Self {
                create_session: std::mem::transmute(get(0x0694_D52E, "NvAPI_DRS_CreateSession")?),
                destroy_session: std::mem::transmute(get(0xDAD9_CFF8, "NvAPI_DRS_DestroySession")?),
                load_settings: std::mem::transmute(get(0x375D_BD6B, "NvAPI_DRS_LoadSettings")?),
                save_settings: std::mem::transmute(get(0xFCBC_7E14, "NvAPI_DRS_SaveSettings")?),
                find_profile_by_name: std::mem::transmute(get(0x7E4A_9A0B, "NvAPI_DRS_FindProfileByName")?),
                find_application_by_name: std::mem::transmute(get(0xEEE5_66B2, "NvAPI_DRS_FindApplicationByName")?),
                create_profile: std::mem::transmute(get(0xCC17_6068, "NvAPI_DRS_CreateProfile")?),
                delete_profile: std::mem::transmute(get(0x1709_3206, "NvAPI_DRS_DeleteProfile")?),
                create_application: std::mem::transmute(get(0x4347_A9DE, "NvAPI_DRS_CreateApplication")?),
                get_setting: std::mem::transmute(get(0x73BF_8338, "NvAPI_DRS_GetSetting")?),
                set_setting: std::mem::transmute(get(0x577D_D202, "NvAPI_DRS_SetSetting")?),
                get_profile_info: std::mem::transmute(get(0x61CD_6FD6, "NvAPI_DRS_GetProfileInfo")?),
            })
        }
    }
}

fn check(status: Status, what: &str) -> Result<()> {
    match status {
        OK => Ok(()),
        INVALID_USER_PRIVILEGE => bail!(
            "{what}: the driver refuses to write settings without administrator rights (-137); \
             run this once from an administrator prompt, later runs need none"
        ),
        other => bail!("{what} failed: NVAPI status {other}"),
    }
}

/// A DRS session with the driver's settings loaded.
struct Session<'a> {
    api: &'a Api,
    handle: Handle,
}

impl<'a> Session<'a> {
    fn open(api: &'a Api) -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        // SAFETY: an out pointer to a handle.
        check(unsafe { (api.create_session)(&mut handle) }, "NvAPI_DRS_CreateSession")?;
        let session = Self { api, handle };
        // SAFETY: a live session.
        check(unsafe { (api.load_settings)(handle) }, "NvAPI_DRS_LoadSettings")?;
        Ok(session)
    }

    fn find_profile(&self, name: &str) -> Result<Option<Handle>> {
        let name = unicode(name);
        let mut profile = std::ptr::null_mut();
        // SAFETY: a live session, a NUL-terminated name, an out handle.
        match unsafe { (self.api.find_profile_by_name)(self.handle, name.as_ptr(), &mut profile) } {
            OK => Ok(Some(profile)),
            PROFILE_NOT_FOUND => Ok(None),
            other => check(other, "NvAPI_DRS_FindProfileByName").map(|()| None),
        }
    }

    /// The profile the driver gives `exe`, and whether it is [`PROFILE`].
    fn profile_of(&self, exe: &str) -> Result<Option<(Handle, bool)>> {
        let name = unicode(exe);
        let mut profile = std::ptr::null_mut();
        let mut app = blank::<ApplicationV1>(|a| &mut a.version);
        // SAFETY: a live session, a NUL-terminated name, out handle and struct.
        let found = unsafe { (self.api.find_application_by_name)(self.handle, name.as_ptr(), &mut profile, &mut *app) };
        match found {
            OK => Ok(Some((profile, self.find_profile(PROFILE)? == Some(profile)))),
            EXECUTABLE_NOT_FOUND => Ok(None),
            other => check(other, "NvAPI_DRS_FindApplicationByName").map(|()| None),
        }
    }

    /// A profile's name, and whether the driver ships it (predefined).
    fn profile_name(&self, profile: Handle) -> Result<(String, bool)> {
        let mut info = blank::<ProfileV1>(|p| &mut p.version);
        // SAFETY: a live session and profile, an out struct.
        check(unsafe { (self.api.get_profile_info)(self.handle, profile, &mut *info) }, "NvAPI_DRS_GetProfileInfo")?;
        let len = info.name.iter().position(|&u| u == 0).unwrap_or(UNICODE);
        Ok((String::from_utf16_lossy(&info.name[..len]), info.is_predefined != 0))
    }

    fn create_profile(&self, name: &str) -> Result<Handle> {
        let mut info = blank::<ProfileV1>(|p| &mut p.version);
        info.name = *unicode(name);
        let mut profile = std::ptr::null_mut();
        // SAFETY: a live session, a filled profile description, an out handle.
        check(unsafe { (self.api.create_profile)(self.handle, &mut *info, &mut profile) }, "NvAPI_DRS_CreateProfile")?;
        Ok(profile)
    }

    fn add_application(&self, profile: Handle, exe: &str) -> Result<()> {
        let mut app = blank::<ApplicationV1>(|a| &mut a.version);
        app.name = *unicode(exe);
        app.friendly_name = *unicode(exe);
        // SAFETY: a live session and profile, a filled description.
        check(unsafe { (self.api.create_application)(self.handle, profile, &mut *app) }, "NvAPI_DRS_CreateApplication")
    }

    fn get(&self, profile: Handle) -> Result<Option<u32>> {
        let mut setting = blank::<SettingV1>(|s| &mut s.version);
        // SAFETY: a live session and profile, an out struct.
        match unsafe { (self.api.get_setting)(self.handle, profile, SETTING, &mut *setting) } {
            OK => Ok(Some(setting.current[0])),
            SETTING_NOT_FOUND => Ok(None),
            other => check(other, "NvAPI_DRS_GetSetting").map(|()| None),
        }
    }

    fn set(&self, profile: Handle, value: u32) -> Result<()> {
        let mut setting = blank::<SettingV1>(|s| &mut s.version);
        setting.id = SETTING;
        setting.kind = 0; // NVDRS_DWORD_TYPE
        setting.current[0] = value;
        // SAFETY: a live session and profile, a filled setting.
        check(unsafe { (self.api.set_setting)(self.handle, profile, &mut *setting) }, "NvAPI_DRS_SetSetting")
    }

    fn delete_profile(&self, profile: Handle) -> Result<()> {
        // SAFETY: a live session and one of its profiles.
        check(unsafe { (self.api.delete_profile)(self.handle, profile) }, "NvAPI_DRS_DeleteProfile")
    }

    fn save(&self) -> Result<()> {
        // SAFETY: a live session.
        check(unsafe { (self.api.save_settings)(self.handle) }, "NvAPI_DRS_SaveSettings")
    }
}

impl Drop for Session<'_> {
    fn drop(&mut self) {
        // SAFETY: the session this value opened, destroyed once.
        let _ = unsafe { (self.api.destroy_session)(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structs_have_nvapis_sizes() {
        assert_eq!(size_of::<ProfileV1>(), 4116);
        assert_eq!(size_of::<ApplicationV1>(), 12296);
        assert_eq!(size_of::<SettingV1>(), 12320);
        assert_eq!(version_1::<SettingV1>(), 0x0001_3020);
    }

    #[test]
    fn the_reserve_is_bytes_and_fits_a_dword() {
        assert_eq!(reserve_bytes(512), 0x2000_0000);
        assert_eq!(reserve_bytes(1024), 0x4000_0000);
        assert_eq!(reserve_bytes(2048), 0x8000_0000);
    }

    #[test]
    fn names_are_cut_and_terminated() {
        let name = unicode("victory.exe");
        assert_eq!(&name[..11], &"victory.exe".encode_utf16().collect::<Vec<_>>()[..]);
        assert_eq!(name[11], 0);
        assert_eq!(unicode(&"x".repeat(5000))[UNICODE - 1], 0);
    }
}
