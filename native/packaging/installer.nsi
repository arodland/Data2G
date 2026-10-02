; The Windows installer, lifted from SSTVAE. Driven by tools/make_installer.sh:
;
;   makensis -DVERSION=x.y.z -DSRCDIR=<staged tree> -DOUTFILE=<setup.exe> installer.nsi
;
; NSIS rather than WiX: what is installed is "whatever windeployqt decided",
; which changes with the Qt version, so the honest description is a
; directory (File /r), not an MSI component list. Per-machine, so admin.

Unicode true
!include "MUI2.nsh"
!include "x64.nsh"
!include "LogicLib.nsh"
!include "FileFunc.nsh"

!define UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\Data2G"

!ifndef VERSION
  !error "VERSION is required (see tools/make_installer.sh)"
!endif
!ifndef SRCDIR
  !error "SRCDIR is required (see tools/make_installer.sh)"
!endif
!ifndef OUTFILE
  !error "OUTFILE is required (see tools/make_installer.sh)"
!endif

; The Start Menu entry opens the GUI when the build has one; a host-only
; build gets a shortcut that opens a console running data2g-host.
!if /FileExists "${SRCDIR}\data2g-gui.exe"
  !define MAINEXE "data2g-gui.exe"
!else
  !define MAINEXE "data2g-host.exe"
!endif

Name "Data2G ${VERSION}"
OutFile "${OUTFILE}"
InstallDir "$PROGRAMFILES64\Data2G"
InstallDirRegKey HKLM "Software\Data2G" "InstallDir"
RequestExecutionLevel admin
SetCompressor /SOLID lzma

VIProductVersion "${VERSION}.0"
; Same placeholder publisher as data2g.rc.in; both must match the signing
; certificate's subject once there is one.
VIAddVersionKey "CompanyName" "Data2G"
VIAddVersionKey "ProductName" "Data2G"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "FileVersion" "${VERSION}.0"
VIAddVersionKey "FileDescription" "Data2G installer"

!define MUI_ICON "${__FILEDIR__}\data2g.ico"
!define MUI_UNICON "${__FILEDIR__}\data2g.ico"
!define MUI_ABORTWARNING

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
; Launched through explorer.exe so the app gets the user's token, not this
; elevated installer's (SSTVAE: wrong LOCALAPPDATA, UIPI-blocked drops).
!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_TEXT "Run Data2G"
!define MUI_FINISHPAGE_RUN_FUNCTION LaunchAsUser
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

Section "Data2G (required)" SecMain
  SectionIn RO
  SetOutPath "$INSTDIR"
  ; `\*`, not `\*.*`: the latter skips files without an extension.
  File /r "${SRCDIR}\*"

  WriteRegStr HKLM "Software\Data2G" "InstallDir" "$INSTDIR"

  CreateDirectory "$SMPROGRAMS\Data2G"
  CreateShortcut "$SMPROGRAMS\Data2G\Data2G.lnk" "$INSTDIR\${MAINEXE}"
  CreateShortcut "$SMPROGRAMS\Data2G\Uninstall Data2G.lnk" "$INSTDIR\uninstall.exe"

  WriteRegStr HKLM "${UNINST_KEY}" "DisplayName" "Data2G"
  WriteRegStr HKLM "${UNINST_KEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKLM "${UNINST_KEY}" "DisplayIcon" "$INSTDIR\${MAINEXE}"
  WriteRegStr HKLM "${UNINST_KEY}" "Publisher" "Data2G"
  WriteRegStr HKLM "${UNINST_KEY}" "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegStr HKLM "${UNINST_KEY}" "QuietUninstallString" '"$INSTDIR\uninstall.exe" /S'
  WriteRegStr HKLM "${UNINST_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegDWORD HKLM "${UNINST_KEY}" "NoModify" 1
  WriteRegDWORD HKLM "${UNINST_KEY}" "NoRepair" 1
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD HKLM "${UNINST_KEY}" "EstimatedSize" "$0"

  WriteUninstaller "$INSTDIR\uninstall.exe"
SectionEnd

Section /o "Desktop shortcut" SecDesktop
  CreateShortcut "$DESKTOP\Data2G.lnk" "$INSTDIR\${MAINEXE}"
SectionEnd

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecMain} "The application, Qt and Hamlib."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecDesktop} "A desktop shortcut as well as the Start Menu entry."
!insertmacro MUI_FUNCTION_DESCRIPTION_END

Function .onInit
  ${IfNot} ${RunningX64}
    MessageBox MB_ICONSTOP "Data2G needs 64-bit Windows."
    Abort
  ${EndIf}
  SetRegView 64
FunctionEnd

Function LaunchAsUser
  SetOutPath "$INSTDIR"
  Exec '"$WINDIR\explorer.exe" "$INSTDIR\${MAINEXE}"'
FunctionEnd

Section "Uninstall"
  SetRegView 64
  Delete "$DESKTOP\Data2G.lnk"
  Delete "$SMPROGRAMS\Data2G\Data2G.lnk"
  Delete "$SMPROGRAMS\Data2G\Uninstall Data2G.lnk"
  RMDir "$SMPROGRAMS\Data2G"
  ; Guarded: an empty $INSTDIR would make this delete the drive root.
  ${If} $INSTDIR != ""
    RMDir /r "$INSTDIR"
  ${EndIf}
  DeleteRegKey HKLM "${UNINST_KEY}"
  DeleteRegKey HKLM "Software\Data2G"
SectionEnd
