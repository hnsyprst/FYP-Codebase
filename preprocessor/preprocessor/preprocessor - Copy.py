import os
import librosa
import librosa.display
import math
import json
import matplotlib.pyplot as plt

datasetPath = r"D:\Work\UNI\!-FinalYear\FYP\ExtremelyLimitedDataset"
jsonPath = r"D:\Work\UNI\!-FinalYear\FYP\ExtremelyLimitedDataset\data.json"

# All files will be loaded at this sample rate
sampleRateConst = 22050
# All files will be this long (in seconds)
durationConst = 30
# Based on the preceeding constants, each file will have this many samples
samplesPerFileConst = sampleRateConst * durationConst

def saveMFCC(datasetPath, jsonPath,
             numMFCCs=13, numFFT=2048, hopLength=512, numSegments=5):
    progressData = {
        "Genre":        [],
        "NumFiles":     [],
        "TotalFiles":   0
        }
    
    for thisDirectoryPath, thisDirectory, filenames in os.walk(datasetPath):
        # If we are in the root directory
        if thisDirectoryPath is datasetPath:
            # Fill the Mapping list in the data dictionary with the list of subfolder names
            progressData["Genre"] = thisDirectory
        else:
            numFilesInDirectory = 0
            # Process the audio files in this genre subfolder
            for file in filenames:
                numFilesInDirectory += 1
            progressData["NumFiles"].append(numFilesInDirectory)
            progressData["TotalFiles"] += numFilesInDirectory
    print(progressData)

    # This dictionary will store the output data
    data = {
        "Mapping":  [],
        "MFCC":     [],
        "Labels":   []
        }

    numFilesCompleted = 0
    
    # Each segment of each file is expected to have this many samples
    numSamplesPerSegment = int(samplesPerFileConst / numSegments)
    # This variable will be used to ensure the shape of all generated MFCCs is the same;
    # some files may have slightly more or less samples per segment than expected due to
    # slight fluctuations in overall duration. math.ceil ensures any floating point values
    # returned will be rounded up.
    expectedNumMFCCVectorsPerSegment = math.ceil(numSamplesPerSegment / hopLength)

    # This loop will iterate through all the files in each genre
    for i, (thisDirectoryPath, thisDirectory, filenames) in enumerate(os.walk(datasetPath)):
        # If we are in the root directory
        if thisDirectoryPath is datasetPath:
            # Fill the Mapping list in the data dictionary with the list of subfolder names
            data["Mapping"] = thisDirectory
        else:
            print("Processing: " + thisDirectoryPath)
            # Process the audio files in this genre subfolder
            for file in filenames:
                filePath = os.path.join(thisDirectoryPath, file)
                signal, sampleRate = librosa.load(filePath, sampleRateConst)
                
                fig, ax = plt.subplots(1)
                fig.suptitle("Mella Dee - Donny's Groove")
                for segment in range(numSegments):
                    #plt.figure(figsize=(16, 8))


                    # The first sample in this segment
                    #   (Multiplying the number of samples per segment by the current segment number
                    #   gives the start position of the current segment)
                    startSample = numSamplesPerSegment * segment
                    # The final sample in this segment
                    finishSample = startSample + numSamplesPerSegment

                    # Create MFCC sequence for this segment
                    mfcc = librosa.feature.mfcc(signal[startSample:finishSample],
                                                sr=sampleRate,
                                                n_mfcc = numMFCCs,
                                                n_fft = numFFT,
                                                hop_length = hopLength)

                    if segment == 4:
                        im = librosa.display.specshow(mfcc, x_axis='s', sr=sampleRateConst, ax=ax)
                        ax.set_ylabel(f'MFCC {segment}')
                        plt.colorbar(im, ax=ax)
                                            

                    # Perform a matrix transpose on the MFCC (swap the rows and columns of the matrix)
                    mfcc = mfcc.T
                    
                    # Commit the generated MFCC sequence to the dictionary only if it has the
                    # expected shape
                    if len(mfcc) == expectedNumMFCCVectorsPerSegment:
                        data["MFCC"].append(mfcc.tolist())
                        data["Labels"].append(i - 1)
                        print("Finished: " + file + ", segment: " + str(segment))
                
                numFilesCompleted += 1

                #fig.colorbar()
                plt.tight_layout()
                plt.show()

                percentCompleted = (numFilesCompleted / progressData["TotalFiles"]) * 100
                completeString = f'Complete: {numFilesCompleted} ({percentCompleted}%)'
                print(completeString)

    with open(jsonPath, "w") as jsonOutputFile:
        json.dump(data, jsonOutputFile, indent = 4)

def visualise(mfcc):
    # Perform a matrix transpose on the MFCC (swap the rows and columns of the matrix)
    plt.figure(figsize=(16, 8))
    plt.title('24 Hour Experience - Together')
    librosa.display.specshow(mfcc, x_axis='s', sr=sampleRateConst)
    plt.ylabel('MFCC')
    plt.colorbar()

    plt.show()

if __name__ == "__main__":
    saveMFCC(datasetPath, jsonPath)
    print("Completed script.")