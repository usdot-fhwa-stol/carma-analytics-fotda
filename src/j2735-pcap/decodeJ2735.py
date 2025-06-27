import J2735_201603_2023_06_22
import sys
import json
import csv
import pyshark
import os
from binascii import unhexlify
from collections import defaultdict

current_directory = os.getcwd()
parent_directory = os.path.abspath(os.path.join(current_directory, os.path.pardir))
decodedDir= current_directory + '/decoded'

capture = pyshark.FileCapture(sys.argv[1])

def formatFileName():
    fileList = sys.argv[1].split('/')
    file = fileList[-1]
    fileName = 'decoded_' + file.replace('.pcap', '.txt')
    return fileName

def formatFileName_csv():
    fileList = sys.argv[1].split('/')
    file = fileList[-1]
    fileName_csv = 'decoded_' + file.replace('.pcap', '.csv')
    return fileName_csv

def readLines():
    fileList = sys.argv[1].split('/')
    fileList[-1] = "pcap.txt"
    inputFile = "/".join(fileList)
    f = open(inputFile, 'r')
    Lines = f.readlines()
    f.close()

    return Lines

def writeIds(w, msgId_count):
    w.write('\nDecoded Message ID Counts:\n')
    print('\nDecoded Message ID Counts:')
    for msgId, count in msgId_count.items():
        w.write(f'{msgId}: {count}\n')
        print(f'{msgId}: {count}')

def isValidMessage(line):
    tempFrame = line[6:]
    if (len(tempFrame.strip('\n')) > 510):
        frameSize = 8
        encodedSize = int(line[5:8], 16) * 2
    else: 
        frameSize = 6
        encodedSize = int(line[4:6], 16) * 2

    newFrame = line[frameSize:].strip('\n')
    if (encodedSize == len(newFrame)):
        return True
    else:
        return False
    
def convertBytes(obj):
    if isinstance(obj, dict):
        return {k: convertBytes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convertBytes(item) for item in obj]
    elif isinstance(obj, tuple):
        return [convertBytes(item) for item in obj]
    elif isinstance(obj, bytes):
        return obj.hex()
    else:
        return obj

def decode(data, frame, w, msgId_count, id, csv_output, i):
    #Write data to text file
    w.write(data)
    w.write('\n')

    frame.from_uper(unhexlify(data))
    cleanObj = convertBytes(frame())
    jsonString = json.dumps(cleanObj, indent=2)
    jsonDict = json.loads(jsonString)
    message_type = jsonDict['value'][0]
    data_value = data
    #Write data to the CSV file
    csv_writer = csv.writer(csv_output)
    packet_number = i
    packet_count = capture[i]
    timestamp = packet_count.sniff_time
    if i==1:
        # Write the CSV header
        csv_writer.writerow(['Packet Number', 'Timestamp', 'Message Type', 'Payload'])
    csv_writer.writerow([packet_number, timestamp, message_type, data_value])

    #Write json string to text file
    w.write(jsonString)
    w.write('\n')

    msgId_count[id] += 1  # increment count for successfully decoded msgId

def main():
    frame = J2735_201603_2023_06_22.DSRC.MessageFrame
    msgIds = ['0012','0013','0014','001f','0020','0029'] # can be updated to include other PSIDs
    msgId_count = defaultdict(int)  # dictionary to track decoded msgId and their counts
    successCount = 0
    fileName = formatFileName()
    fileName_csv = formatFileName_csv()
    w = open(decodedDir + '/' + fileName, 'w')
    csv_output = open(decodedDir + '/' + fileName_csv, 'w', newline='')
    i=1 # Iterator for packet number
    packet_total_count = 0
    for line in readLines():
        packet_total_count += 1
        for id in msgIds:
            idx = line.find(id)
            if (idx != -1):
                data = line[idx:].strip('\n')
                if (isValidMessage(data) == True):
                    try:
                        decode(data, frame, w, msgId_count, id, csv_output, i)
                        i += 1
                        successCount += 1
                    except Exception as e:
                        print("Error decoding message: ", e)
                        continue
            else: 
                continue

    # Write the decoded message IDs and their counts to the output file
    writeIds(w, msgId_count)
    w.close()
    csv_output.close()
    errorCount = packet_total_count - successCount

    print(f"Decoding of file {fileName} Complete.  Successfully decoded {successCount} msgs, Failed to decode {errorCount}")
    sys.exit(0)

if __name__=="__main__":
    main()
