from Player import Player
from params import *
import numpy as np


# the losses of the rhythms
def rhythm_diff(newRhythm, lastRythm):
    if len(newRhythm) == len(lastRythm) and \
            np.sum(np.array(lastRythm) == np.array(newRhythm)) == len(lastRythm): return np.inf
    tt = min(len(lastRythm), len(newRhythm))
    a = np.array(newRhythm)
    a = a[a!=0]
    b=np.array(lastRythm)
    b=b[b!=0]
    return np.sum(np.abs(a[:tt] - np.flip(b[-tt:], axis=0)) * np.arange(tt, 0, -1))

def rhythm_diff2(newRhythm, lastRythm):
    if len(newRhythm) == len(lastRythm) and \
            np.sum(np.array(lastRythm) == np.array(newRhythm)) == len(lastRythm): return np.inf
    tt = min(len(lastRythm), len(newRhythm))
    a = np.array(newRhythm)
    a = a[a!=0]
    b=np.array(lastRythm)
    b=b[b!=0]
    return np.sum(np.abs(a[:tt] - b[:tt]))


if __name__ == '__main__':

    from copy import copy
    vel2 = []
    for v in velocities:
        temp=[]
        for vv in v:
            if vv>30:
                temp.append(vv-25)
            else:
                temp.append(vv)
        vel2.append(copy(temp))
    # nvel = np.array(narativeVelocity)
    nvel = [vv-25 if vv>30 else vv for vv in narativeVelocityJazz]
    # vel3 = [list(v.astype(int)) for v in vel2]
	
	
    # the folder of the data midi files
    folderName = "midiToLearn"
    
    # create a new John Chain!!!!!!!
    # folderName and rhythms are defined in params.py
    player = Player(folderName, rhythms, xNotes=3, lowestNote=52, highestNote=76,
                    scaleDict=dict1, velocities=vel2, narativeRhythm= narativeRhythmJazz,
                    narativeVelocity=nvel, narrativeNotes = narrativeNotes,
                    rhythmDiffFunc = rhythm_diff2, uniformExplorationRateBetween = 0.2,
                    uniformExplorationRateWithin = 0.16, closeExplorationRateBetween = 0.1,
                    closeExplorationRateWithin = 0.0,  narrativeRateBetween = 0,
                    narrativeRateWithin = 0,    scaleHeuristic = [
                    [36, 48, 60, 72, 84],
                    [40, 43, 52, 55, 64, 67, 76, 79, 88, 91],
                    [38, 45, 50, 57, 62, 69, 74, 81, 86]],
                    scaleHeuristicP = [1.5, 1.3, 1.1])

    player.createSong(ticks_per_beat=480) # start a new song file
    base_scale = -11 #set the scale of the song
    # add the channels of the songs
    player.add_track(startingNotes = [48, 52, 55], app_notes=base_scale,returnRhytem=0.3, narativeP=0.2, instrument=2, pChords =0.3, q=[1.0/8,1.0/4,1.0/4,1.0/8,1.0/8,1.0/16,1.0/16], timeLine=np.ones(32))
    player.add_track_same(app_notes=base_scale+24, instrument=2, pChords = 0, q=[1.0/8,1.0/4,1.0/4,1.0/8,1.0/8,1.0/16,1.0/16], timeLine=[1,1,0,0]*8)
    player.add_track2(startingNotes = [48, 52, 55], app_notes=base_scale+12, instrument=60, pChords = 0.3, q=[1.0/8,1.0/4,1.0/4,1.0/8,1.0/8,1.0/16,1.0/16], timeLine=[0,0,0,0,0,0,1,1]*4)

    # saving the song
    player.save_creation("JohnChain_TheOneAndOnly")
