import subprocess
import os

def ffmpeg_trim(filename, fileIn, fileOut, trimPercent, trimDuration):
    # ffprobe used as a subprocess to return the duration of the input song in seconds
    # ffprobe settings:
    #   -v quiet
    #       displays no log - no errors, no warnings, no debugging, no processing info
    #   -i
    #       opens the file at the path given after -i
    #   -show_entries format=duration
    #       used to only display metadata pertaining to the duration of the song
    #   -of csv="p=0"
    #       sets the output format to csv
    #       "p=0" disables printing the section name at the beginning of each line
    # together these settings ensure only the duration is returned
    fileDuration = float(subprocess.getoutput('ffprobe -v quiet -i "' + fileIn + '" -show_entries format=duration -of csv="p=0"'))

    trimStart = fileDuration * trimPercent
    trimEnd = trimStart + trimDuration

    # ffmpeg used as a subprocess to trim the file
    # the song is trimmed starting at the given percentage of the way through
    # and ending the given number of seconds later
    # ffmpeg settings:
    #   -hide_banner
    #       hides the ffmpeg banner - this makes the console output for each file smaller so it is easier to read
    #   -n
    #       disables overwriting - this saves time trimming files that have already been done in case of a crash
    #   -i
    #       opens the file at the path given after -i
    #   -ss
    #       seeks the input file to the time (in seconds) given after -ss 
    #   -to
    #       stops writing the output file after the given number of seconds
    #   -b:a 320k
    #       sets the audio bitrate to 320kbps
    cmdOutput = subprocess.run('ffmpeg -hide_banner -n -i "' + str(fileIn) + '" -ss ' + str(trimStart) + ' -to ' + str(trimEnd) + ' -b:a 320k "' + fileOut + '"',
                               text=True)

# directory containing the songs to be trimmed
inDirectory = "F:/Roy's Phat Tunes/UK Hardcore/"
# output directory
outDirectory = "D:/Work/UNI/!-Final Year/FYP/Dataset/UK Hardcore/"
# each song in the directory will be trimmed starting this percent of the way through the song
globalTrimPercent = 0.25
# and ending this many seconds later
globalTrimDuration = 30

for filename in os.listdir(inDirectory):
    inPath = inDirectory + filename
    # the extension is removed from each filename and replaced with .mp3
    # this means that any input files not already in mp3 format will be converted by ffmpeg during the trimming
    outPath = outDirectory + os.path.splitext(filename)[0] + '.mp3'

    ffmpeg_trim(filename, inPath, outPath, globalTrimPercent, globalTrimDuration)