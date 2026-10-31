import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Data2G

// Two screens behind a bottom tab bar: the modem (waterfall, level, start and
// stop, activity log) and its settings (callsign, audio, rig).
ApplicationWindow {
    id: window
    visible: true
    title: "Data2G TNC"

    Tnc {
        id: tnc
        onError: function(message) {
            errorText.text = message
            errorDialog.open()
        }
    }
    RigSettings { id: rigSettings }

    // Back while the modem runs sends the app to the background: ending the
    // activity would end the process, and the modem with it. Stopped, Back
    // leaves as Android users expect.
    onClosing: function(close) {
        if (tnc.running) {
            tnc.moveToBackground()
            close.accepted = false
        }
    }

    // Edge-to-edge (targetSdk 35+): inset from the system bars ourselves.
    header: ToolBar {
        topPadding: SafeArea.margins.top
        contentItem: RowLayout {
            Label {
                text: "Data2G TNC"
                font.bold: true
                font.pixelSize: 18
                Layout.leftMargin: 12
            }
            Item { Layout.fillWidth: true }
            Label {
                text: tnc.ptt ? "TX" : tnc.busy ? "BUSY" : ""
                color: tnc.ptt ? "#c62828" : "#e6a01e"
                font.bold: true
                Layout.rightMargin: 12
            }
        }
    }

    footer: TabBar {
        id: tabs
        bottomPadding: SafeArea.margins.bottom
        TabButton { text: "Modem" }
        TabButton { text: "Settings" }
    }

    StackLayout {
        anchors.fill: parent
        currentIndex: tabs.currentIndex

        // ---- Modem
        ColumnLayout {
            spacing: 6

            Waterfall {
                id: waterfall
                Layout.fillWidth: true
                Layout.preferredHeight: window.height * 0.3
            }
            LevelMeter {
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                peak: waterfall.peak
            }
            RowLayout {
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                Button {
                    text: tnc.starting ? "Starting…" : tnc.running ? "Stop" : "Start"
                    enabled: !tnc.starting
                    highlighted: !tnc.running
                    onClicked: tnc.running ? tnc.stop() : tnc.start()
                }
                Label {
                    Layout.fillWidth: true
                    text: tnc.status
                    wrapMode: Text.Wrap
                }
            }
            Label {
                Layout.leftMargin: 12
                text: tnc.frequency
                visible: text !== ""
                font.pixelSize: 16
            }
            Label {
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                text: tnc.counters
                visible: text !== ""
                color: "#c62828"
                font.pixelSize: 11
                wrapMode: Text.Wrap
            }

            RowLayout {
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                Label { text: "Activity"; font.bold: true; Layout.fillWidth: true }
                ToolButton { text: "Clear"; onClicked: tnc.clearLog() }
            }
            ListView {
                id: log
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                clip: true
                model: tnc.log
                ScrollBar.vertical: ScrollBar {}
                delegate: Label {
                    required property string modelData
                    width: log.width
                    text: modelData
                    font.family: "monospace"
                    font.pixelSize: 11
                    wrapMode: Text.WrapAnywhere
                    color: modelData.indexOf(" ERROR: ") >= 0 ? "#c62828"
                         : modelData.indexOf(" WARNING: ") >= 0 ? "#ef6c00" : palette.text
                }
                // Follow the newest line while scrolled to the end.
                onCountChanged: if (atYEnd || contentHeight <= height) Qt.callLater(positionViewAtEnd)
            }
        }

        // ---- Settings
        ScrollView {
            id: settingsView
            contentWidth: availableWidth

            ColumnLayout {
                width: settingsView.availableWidth - 24
                x: 12
                spacing: 8

                Label {
                    Layout.fillWidth: true
                    Layout.topMargin: 8
                    text: "Changes take effect the next time the modem starts."
                    visible: tnc.running
                    wrapMode: Text.Wrap
                    color: "#ef6c00"
                }

                Label { text: "Station"; font.bold: true; Layout.topMargin: 8 }
                Label { text: "Callsign" }
                TextField {
                    Layout.fillWidth: true
                    text: tnc.callsign
                    inputMethodHints: Qt.ImhUppercaseOnly | Qt.ImhNoPredictiveText
                    onTextEdited: tnc.callsign = text
                }

                Label { text: "Audio"; font.bold: true; Layout.topMargin: 12 }
                Label { text: "Input (from the radio)" }
                ComboBox {
                    Layout.fillWidth: true
                    model: tnc.inputDevices
                    function sync() { currentIndex = Math.max(0, model.indexOf(tnc.inputDevice)) }
                    Component.onCompleted: sync()
                    onModelChanged: sync()
                    onActivated: tnc.inputDevice = currentText
                }
                Label { text: "Output (to the radio)" }
                ComboBox {
                    Layout.fillWidth: true
                    model: tnc.outputDevices
                    function sync() { currentIndex = Math.max(0, model.indexOf(tnc.outputDevice)) }
                    Component.onCompleted: sync()
                    onModelChanged: sync()
                    onActivated: tnc.outputDevice = currentText
                }
                Button { text: "Refresh devices"; onClicked: tnc.refreshDevices() }
                Label { text: "Transmit level: " + tnc.txLevelDb.toFixed(0) + " dB" }
                Slider {
                    Layout.fillWidth: true
                    from: -30; to: 0; stepSize: 1
                    value: tnc.txLevelDb
                    onMoved: tnc.txLevelDb = value
                }
                Label {
                    Layout.fillWidth: true
                    wrapMode: Text.Wrap
                    font.pixelSize: 11
                    opacity: 0.7
                    text: "Set the radio's input so the transmit level here stays near 0 dB "
                          + "without ALC action. The receive level is good anywhere in the green."
                }

                Label { text: "Rig control"; font.bold: true; Layout.topMargin: 12 }
                RigPane {
                    Layout.fillWidth: true
                    rig: rigSettings
                }

                Label { text: "Connecting a client"; font.bold: true; Layout.topMargin: 12 }
                Label {
                    Layout.fillWidth: true
                    wrapMode: Text.Wrap
                    text: "KISS on port " + tnc.kissPort + ", VARA HF-style commands on port "
                          + tnc.varaPort + " (data on " + (tnc.varaPort + 1) + "), at 127.0.0.1 from this phone"
                          + (tnc.listenAll && tnc.addresses.length > 0
                             ? ", or " + tnc.addresses.join(" or ") + " from the network." : ".")
                }
                Switch {
                    text: "Accept connections from the network"
                    checked: tnc.listenAll
                    onToggled: tnc.listenAll = checked
                }
                Label {
                    Layout.fillWidth: true
                    wrapMode: Text.Wrap
                    font.pixelSize: 11
                    opacity: 0.7
                    text: "Listens on every interface (0.0.0.0) instead of this phone only. Anyone "
                          + "who can reach the phone can then use the TNC, and transmit with your "
                          + "callsign: use it only on a network you trust."
                }
                Label {
                    Layout.fillWidth: true
                    Layout.bottomMargin: 16
                    font.pixelSize: 11
                    opacity: 0.7
                    text: "Data2G " + Application.version
                }
            }
        }
    }

    Dialog {
        id: errorDialog
        parent: Overlay.overlay
        anchors.centerIn: parent
        width: Math.min(parent ? parent.width - 48 : 300, 420)
        modal: true
        title: "Data2G TNC"
        standardButtons: Dialog.Ok
        Label {
            id: errorText
            width: parent.width
            wrapMode: Text.Wrap
        }
    }
}
