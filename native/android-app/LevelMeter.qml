import QtQuick

// The input level as a bar in dBFS (SSTVAE's LevelMeter): linear would
// squash everything an operator tunes by into the leftmost pixels. Three
// zones: too quiet (orange), good (green, deliberately wide), clipping (red).
Item {
    id: root

    property real peak: 0.0  // 0..1 linear

    implicitHeight: 26

    // Qt colours are #AARRGGBB, not CSS's #RRGGBBAA.
    readonly property real floorDb: -60
    readonly property real db: peak > 0 ? 20 * Math.log(peak) / Math.LN10 : floorDb
    readonly property real frac: Math.max(0, Math.min(1, (db - floorDb) / (0 - floorDb)))

    function xFor(d) { return width * Math.max(0, (d - floorDb) / (0 - floorDb)) }

    Rectangle {
        anchors.fill: parent
        radius: 3
        color: "#14000000"
        border.color: "#22000000"
    }
    Rectangle {  // the good band, behind the bar
        x: root.xFor(-30)
        width: root.xFor(-6) - x
        height: parent.height
        color: "#182e7d32"
    }
    Rectangle {
        x: 1; y: 1
        height: parent.height - 2
        width: Math.max(0, root.frac * (parent.width - 2))
        radius: 2
        color: root.db > -3 ? "#c62828" : root.db > -30 ? "#2e7d32" : "#ef6c00"
        Behavior on width { NumberAnimation { duration: 80 } }
    }
    Text {
        anchors.centerIn: parent
        font.pixelSize: 11
        color: "#aa000000"
        text: root.peak <= 0 ? "No audio"
            : (root.db > -3 ? "Too loud" : root.db > -30 ? "Level good" : "Quiet")
              + "  " + root.db.toFixed(0) + " dBFS"
    }
}
