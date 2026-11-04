#!/bin/bash
if ! command -v tshark >/dev/null 2>&1; then
    echo "TShark is not installed. Installing TShark..."
    sudo apt install -y tshark
else
    echo "TShark is already installed."
fi

pip install -r requirements.txt

currentDir="$PWD"
logsDir=$currentDir/logs

if [ ! -d "$currentDir/decoded" ]; then
  mkdir "$currentDir/decoded"
else
  echo "Directory '$currentDir/decoded' already exists."
fi

if [ ! -d "$currentDir/logs" ]; then
  mkdir "$currentDir/logs"
else
  echo "Directory '$currentDir/logs' already exists."
fi

extract() {
    echo ""
    ls *.pcap
    read -rep "Type pcap file from list: " fileName

    tshark -r "$fileName" --disable-protocol wsmp -Tfields -Eseparator=, -e data.data > pcap.txt
}

decode() {
    python3 decodeJ2735.py "$currentDir"/"$fileName"
    cd "$currentDir" || exit
    mv pcap.txt "$logsDir"
    mv -- *.pcap "$logsDir"
}

processing() {
    extract
    decode
}

processing
