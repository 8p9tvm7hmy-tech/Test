; Inno Setup script — builds BourseDashboard-Setup.exe from the PyInstaller output.
#define AppVersion GetEnv("APP_VERSION")
#if AppVersion == ""
  #define AppVersion "1.0.0"
#endif

[Setup]
AppId={{8E4B6C1E-5B7A-4C2E-9F11-3A6D2B7C9E01}
AppName=Bourse Dashboard
AppVersion={#AppVersion}
AppPublisher=Bourse Dashboard
DefaultDirName={autopf}\BourseDashboard
DefaultGroupName=Bourse Dashboard
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename=BourseDashboard-Setup
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\BourseDashboard.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; Per-user install: no administrator rights needed.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\BourseDashboard\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Bourse Dashboard"; Filename: "{app}\BourseDashboard.exe"
Name: "{group}\{cm:UninstallProgram,Bourse Dashboard}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Bourse Dashboard"; Filename: "{app}\BourseDashboard.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\BourseDashboard.exe"; Description: "{cm:LaunchProgram,Bourse Dashboard}"; Flags: nowait postinstall skipifsilent
