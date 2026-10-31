import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// Rig control settings (SSTVAE's rig screen, settings only): the modem opens
// the rig when it starts, so there is no Connect button here.
ColumnLayout {
    id: pane
    spacing: 8

    required property var rig

    // USB devices come and go with the cable: re-list whenever this shows.
    onVisibleChanged: if (visible) rig.refreshDevices()

    component Hint: Label {
        font.pixelSize: 11
        opacity: 0.7
        Layout.fillWidth: true
        wrapMode: Text.Wrap
    }

    Switch {
        text: "Control the radio"
        checked: pane.rig.enabled
        onToggled: pane.rig.enabled = checked
    }
    Hint {
        text: "Keys the transmitter over a USB serial adapter, a Bluetooth radio, or a "
              + "station computer on the network. Leave it off to let the radio's VOX key it."
    }

    ColumnLayout {
        Layout.fillWidth: true
        spacing: 8
        visible: pane.rig.enabled

        Label { text: "Connection" }
        ComboBox {
            Layout.fillWidth: true
            textRole: "text"
            valueRole: "value"
            model: [
                { text: "USB serial", value: "usb" },
                { text: "Bluetooth", value: "bluetooth" },
                { text: "Network (rigctld or ser2net)", value: "network" }
            ]
            Component.onCompleted: currentIndex = indexOfValue(pane.rig.connection)
            onActivated: pane.rig.connection = currentValue
        }

        Label { text: "Radio" }
        Button {
            Layout.fillWidth: true
            text: pane.rig.modelLabel
            onClicked: modelDialog.open()
        }

        // ---- a USB or Bluetooth device
        ColumnLayout {
            Layout.fillWidth: true
            spacing: 4
            visible: pane.rig.connection !== "network"

            Label { text: "Device" }
            ComboBox {
                id: deviceCombo
                Layout.fillWidth: true
                model: pane.rig.devices
                textRole: "label"
                valueRole: "id"
                // The index of a device isn't stable across a replug.
                function sync() { currentIndex = indexOfValue(pane.rig.device) }
                Component.onCompleted: sync()
                Connections {
                    target: pane.rig
                    function onDevicesChanged() { deviceCombo.sync() }
                }
                onActivated: pane.rig.device = currentValue
                delegate: ItemDelegate {
                    id: deviceItem
                    required property var modelData
                    required property int index
                    width: deviceCombo.width
                    highlighted: deviceCombo.highlightedIndex === index
                    contentItem: Label { text: deviceItem.modelData.label; wrapMode: Text.Wrap }
                }
            }
            RowLayout {
                Button { text: "Refresh"; onClicked: pane.rig.refreshDevices() }
                Button {
                    text: "Grant access"
                    // Bluetooth asks for the permission to list at all, before any device.
                    visible: !pane.rig.devicePermitted
                             && (pane.rig.device !== "" || (pane.rig.connection === "bluetooth" && !pane.rig.bluetoothReady))
                    onClicked: pane.rig.requestPermission()
                }
            }
            Hint {
                text: pane.rig.connection === "bluetooth" && !pane.rig.bluetoothReady
                      ? "Android needs Bluetooth access before paired radios can be listed."
                      : pane.rig.devices.length === 0
                        ? (pane.rig.connection === "bluetooth"
                           ? "No paired radios. Pair the radio in Android's Bluetooth settings first."
                           : "Nothing plugged in. Connect the radio or its serial adapter, then Refresh.")
                        : pane.rig.devicePermitted ? ""
                        : "Android needs permission before this device can be opened."
                visible: text !== ""
            }

            Label { text: "Speed"; visible: pane.rig.connection === "usb" }
            ComboBox {
                Layout.fillWidth: true
                visible: pane.rig.connection === "usb"
                model: ["Default", "1200", "2400", "4800", "9600", "19200", "38400", "57600", "115200"]
                Component.onCompleted: currentIndex = pane.rig.baud === 0 ? 0 : Math.max(0, model.indexOf(String(pane.rig.baud)))
                onActivated: pane.rig.baud = currentIndex === 0 ? 0 : parseInt(currentText)
            }
        }

        // ---- or a host
        ColumnLayout {
            Layout.fillWidth: true
            spacing: 4
            visible: pane.rig.connection === "network"

            Label { text: "Host" }
            TextField {
                Layout.fillWidth: true
                text: pane.rig.host
                inputMethodHints: Qt.ImhNoAutoUppercase | Qt.ImhNoPredictiveText
                // Per keystroke: Back hides the keyboard without editingFinished.
                onTextEdited: pane.rig.host = text
            }
            Hint {
                text: "With the radio set to \"Hamlib NET rigctl\" this is a computer running "
                      + "rigctld. With a real radio chosen instead, it is a serial-over-network "
                      + "server such as ser2net. For example 192.168.1.10:4532; the port defaults to 4532."
            }
        }

        // ---- keying
        Label { text: "Transmit keying" }
        ComboBox {
            id: pttBox
            Layout.fillWidth: true
            model: pane.rig.pttMethods
            function label(v) {
                switch (v) {
                case "vox": return "VOX (the audio keys the radio)"
                case "cat": return "CAT command"
                case "dtr": return "DTR line"
                case "rts": return "RTS line"
                }
                return v
            }
            displayText: label(currentValue !== undefined ? currentValue : pane.rig.pttMethod)
            delegate: ItemDelegate {
                required property var modelData
                required property int index
                width: pttBox.width
                text: pttBox.label(modelData)
                highlighted: pttBox.highlightedIndex === index
            }
            function sync() { currentIndex = Math.max(0, model.indexOf(pane.rig.pttMethod)) }
            Component.onCompleted: sync()
            onModelChanged: sync()  // Bluetooth narrows the list
            onActivated: pane.rig.pttMethod = currentValue
        }
        ComboBox {
            id: pttAudioBox
            Layout.fillWidth: true
            visible: pane.rig.pttAudioSupported && pane.rig.pttMethod === "cat"
            model: ["mic", "data"]
            displayText: currentValue === "data" ? "Transmit audio: data / rear input" : "Transmit audio: mic / front input"
            Component.onCompleted: currentIndex = Math.max(0, model.indexOf(pane.rig.pttAudio))
            onActivated: pane.rig.pttAudio = currentValue
        }
        ComboBox {
            id: otherLineBox
            Layout.fillWidth: true
            visible: pane.rig.pttMethod === "dtr" || pane.rig.pttMethod === "rts"
            readonly property string line: pane.rig.pttMethod === "dtr" ? "RTS" : "DTR"
            model: ["default", "high", "low"]
            displayText: line + " while connected: " + (currentValue === "high" ? "held high"
                         : currentValue === "low" ? "held low" : "default (high)")
            delegate: ItemDelegate {
                required property var modelData
                required property int index
                width: otherLineBox.width
                text: modelData === "default" ? "Default (high)" : "Held " + modelData
                highlighted: otherLineBox.highlightedIndex === index
            }
            function sync() { currentIndex = Math.max(0, model.indexOf(pane.rig.otherLine)) }
            Component.onCompleted: sync()
            onVisibleChanged: sync()
            onActivated: pane.rig.otherLine = currentValue
        }

        Hint {
            text: pane.rig.error
            visible: text !== ""
            color: "#c62828"
            opacity: 1
        }
    }

    // ---- the radio picker: a search box over Hamlib's several hundred rigs
    Dialog {
        id: modelDialog
        parent: Overlay.overlay
        anchors.centerIn: parent
        width: Math.min(parent ? parent.width - 24 : 320, 480)
        height: Math.min(parent ? parent.height - 80 : 400, 560)
        modal: true
        title: "Radio"
        standardButtons: Dialog.Cancel

        onOpened: {
            search.text = ""
            results.model = pane.rig.findModels("")
            search.forceActiveFocus()
        }

        ColumnLayout {
            anchors.fill: parent
            spacing: 8
            TextField {
                id: search
                Layout.fillWidth: true
                placeholderText: "Search: maker or model"
                inputMethodHints: Qt.ImhNoAutoUppercase | Qt.ImhNoPredictiveText
                onTextChanged: results.model = pane.rig.findModels(text)
            }
            ListView {
                id: results
                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true
                ScrollBar.vertical: ScrollBar {}
                delegate: ItemDelegate {
                    id: modelItem
                    required property var modelData
                    width: results.width
                    contentItem: Label { text: modelItem.modelData.label; wrapMode: Text.Wrap }
                    onClicked: {
                        pane.rig.model = modelItem.modelData.number
                        modelDialog.close()
                    }
                }
            }
        }
    }
}
