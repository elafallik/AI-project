from mido import Message, MidiFile, MidiTrack, MetaMessage
from learn import SongLearner
import numpy as np
import glob
from copy import deepcopy

def rhythm_diff(newRhythm, lastRythm):
    if lastRythm == newRhythm: return np.inf
    tt = min(len(lastRythm), len(newRhythm))
    return np.sum(np.abs(newRhythm[:tt] - np.flip(lastRythm[-tt:], axis=0)) * np.arange(tt, 0, -1))


class Player:

    def __init__(self, folderName, rhythms, velocities,
                 xNotes=2,  lowestNote=30, highestNote=128, scaleDict={},
                 narativeRhythm = [], narativeVelocity = [], narrativeNotes=None,
                 rhythmDiffFunc = None,
                 uniformExplorationRateBetween=0, uniformExplorationRateWithin=0,
                 closeExplorationRateBetween=0, closeExplorationRateWithin=0,
                 narrativeRateBetween=0, narrativeRateWithin=0,
                 scaleHeuristic=[[36,48,60,72,84],
                                 [40,43,52,55,64,67,76,79,88,91],
                                 [38,45,50,57,62,69,74,81,86]],
                 scaleHeuristicP=[2.5,2,1.5]):
        """
        its good to have large range of notes here and narrow then in create track
        :param folderName: where the midi file are
        :param rhythms: the rhythms used in the model
        :param velocities:the velocitys same sive as rhythms
        :param xNotes: hte depth of the markov chain
        :param lowestNote: the lowest note (in midi code)
        :param highestNote:
        :param scaleDict: the dict of the song to learn - see utils
        :param narativeRhythm:
        :param narativeVelocity:
        :param narrativeNotes: to pass to learn
        :param rhythmDiffFunc: the loos function on the rhythms
        ###################################the exploration probability###################################
        :param uniformExplorationRateBetween:
        :param uniformExplorationRateWithin:
        :param closeExplorationRateBetween:
        :param closeExplorationRateWithin:
        :param narrativeRateBetween:
        :param narrativeRateWithin:
        :param scaleHeuristic:how andwhat to scale in the note learning
        :param scaleHeuristicP:
        """
        self.lowestNote = lowestNote
        self.highestNote = highestNote
        if len(scaleHeuristic) == len(scaleHeuristicP):

            addition_P = np.ones(128)
            for scalePos, scale in zip(scaleHeuristic, scaleHeuristicP):
                addition_P[scalePos] = scale

            addition_P = addition_P[lowestNote:highestNote+1]
        else:
            addition_P = np.ones(highestNote - lowestNote +1)

        if rhythmDiffFunc ==None:
            self.rhythm_diff_func = rhythm_diff
        else:
            self.rhythm_diff_func = rhythmDiffFunc

        self.xNotes = xNotes
        self.rhythms = rhythms
        self.velocities = velocities
        self.narativeRhythm = narativeRhythm
        self.narativeVelocity = narativeVelocity

        self.uniformExplorationRateBetween = uniformExplorationRateBetween
        self.uniformExplorationRateWithin = uniformExplorationRateWithin
        self.closeExplorationRateBetween = closeExplorationRateBetween
        self.closeExplorationRateWithin = closeExplorationRateWithin
        self.narrativeRateBetween = narrativeRateBetween
        self.narrativeRateWithin = narrativeRateWithin

        self.hadChord = False
        self.savedNotes=[]
        self.sl = SongLearner(lowestNote, highestNote, notesBack=xNotes,
                              narrativeNotes = narrativeNotes, additionVec = addition_P)

        # learning time 8)
        for midiFile in glob.glob(folderName+"/*.mid"):
            fileName = midiFile[midiFile.rfind('/') + 1:-4]
            if fileName in scaleDict:
                print("learning song:", fileName)
                normlizedNotes = scaleDict[fileName]
                self.sl.learn(midiFile, noteShift=normlizedNotes)
            else:
                normlizedNotes = 0

        self.sl.normalizeMatricies()
        self.rhythms_played = []


    def get_r(self, index):
        """
        return the rhythm, -1 return the narrative
        :param index:
        :return:
        """
        if index ==-1:
            return self.narativeRhythm
        return self.rhythms[index]

    def get_v(self, index):
        """
        return the velocity, -1 return the narrative
        :param index:
        :return:
        """
        if index == -1:
            return self.narativeVelocity
        return self.velocities[index]


    def getRhythm(self, prevRhythms):
        """
        get the next rhythm from the best of the loos function
        :param prevRhythms:
        :return:
        """
        if len(prevRhythms) < 1:
            index = np.random.randint(low=0, high=len(self.rhythms))
            return self.rhythms[index], index

        vel = self.get_v(prevRhythms[-1])
        lastRhythm = np.array(self.get_r(prevRhythms[-1]))
        tep=np.where(np.array(vel) > 0)[0]
        lastRhythm = lastRhythm[np.where(np.array(vel) > 0)[0]]

        dists = []
        for i, rhythm in enumerate(self.rhythms):
            r = np.array(rhythm)
            tt=np.where(np.array(self.get_v(i))>0)[0]
            r = r[np.where(np.array(self.get_v(i))>0)[0]]
            diff = self.rhythm_diff_func(r,lastRhythm)
            dists.append([diff,i])
        dists.sort()
        canadates = dists[:4]
        P=[]
        sumLen = 0
        for c in canadates:
           sumLen+=len(self.get_r(c[1]))
        for c in canadates:
            P.append(len(self.get_r(c[1]))/float(sumLen))
        a = np.random.choice(np.arange(0,len(canadates),1), p=P)
        return self.get_r(canadates[a][1]),canadates[a][1]


    def getNotes(self,prevNotes, velocity, numberOfNotes=4, newArray=True):
        """
        the number of nutes from the markov model
        :param prevNotes:
        :param velocity:
        :param numberOfNotes:
        :param newArray:
        :return:
        """
        if newArray:
            notes = []
        else:
            notes = prevNotes
        startIndex = len(notes)
        for i in range(numberOfNotes):
            if(velocity is not None and velocity[startIndex + i]==0):
                notes.append(40)
                continue
            if i ==0:
                n = self.sl.nextNote(prevNotes[-self.xNotes:],
                                     uniformExplorationRate=self.uniformExplorationRateBetween,
                                     closeExplorationRate=self.closeExplorationRateBetween,
                                     narrativeRate=self.narrativeRateBetween)
            else:
                n = self.sl.nextNote(prevNotes[-self.xNotes:],
                                     uniformExplorationRate=self.uniformExplorationRateWithin,
                                     closeExplorationRate=self.closeExplorationRateWithin,
                                     narrativeRate=self.narrativeRateWithin
                                     )
            prevNotes.append(n)
            prevNotes.pop(0)
            notes.append(n)
        return notes

    def addNote(self, track, note, velocity, time):
        """
        add th enote to the channel
        :param track:
        :param note:
        :param velocity:
        :param time:
        :return:
        """
        track.append(Message('note_on', note=note, velocity=velocity, time=2))
        track.append(Message('note_off', note=note, velocity=velocity, time=time - 2))

    def getAdd(self,note):
        """
        get the notes to create the chord, if the chord not in the scale return false
        :param note:
        :return:
        """
        n_dict = {40:[3,7],41:[4,7],43:[4,7],45:[4,7],48:[4,7],50:[3,7],52:[3,7]}
        while True:
            if (note>=40 and note<=52):
                if note not in n_dict.keys():
                    if note == 46: note +=1
                    return n_dict[note+1], False # the open note not in scale
                return n_dict[note], True #do the cord he isin the Scale
            if note>52:
                note -=12
            else:
                note+=12


    def addSimpleChord(self, track, note, velocity, time):
        """
        add the note to th song, if the note is not good (not in the scale) add a normal note
        :param track:
        :param note:
        :param velocity:
        :param time:
        :return:
        """
        add_ , do_or_not= self.getAdd(note)
        #print(add_,do_or_not)
        if not do_or_not: return self.addNote(track,note,velocity,time)
        velocity = 50
        track.append(Message('note_on', note=note+add_[1], velocity=velocity//3, time=2))
        track.append(Message('note_on', note=note+add_[0], velocity=velocity//3 * 2, time=2))
        track.append(Message('note_on', note=note, velocity=velocity, time=2))
        if  np.random.binomial(1,p=1)==True:
            track.append(Message('note_on', note=note+add_[1]+2, velocity=velocity, time=0))
            track.append(Message('note_off', note=note+add_[1]+2, velocity=velocity, time = time-6))
            track.append(Message('note_off', note=note, velocity=velocity, time=0))
        else:
            track.append(Message('note_off', note=note, velocity=velocity, time=time-6))
        track.append(Message('note_off', note=note+add_[1], velocity=velocity, time = 0))
        track.append(Message('note_off', note=note+add_[0], velocity=velocity, time = 0))

    def addNotes(self, track, notes, rhythm, velocity, ticks_per_box,
                 p=0, q=[1.0/8,1.0/4,1.0/4,1.0/8,1.0/8,1.0/16,1.0/16], addition = 0):
        """
        add a set af notes to the model
        can add chords insted og notes
        :param track:
        :param notes:
        :param rhythm:
        :param velocity:
        :param ticks_per_box:
        :param p:
        :param q:
        :param addition:
        :return:
        """

        doChord = 0
        for r, v, note in zip(rhythm, velocity, notes):

            note += addition
            time=min(int(ticks_per_box * r), int(ticks_per_box))

            if v == 0:
                track.append(Message('note_on', note=0, velocity=0, time=time))
                # track.append(Message('note_off', note=0, velocity=0, time=time))
            elif doChord > 0 :
                self.addSimpleChord(track, note, v,time)
                doChord-=1
            elif doChord < -5 and time >= ticks_per_box * 1.0/8 and np.random.binomial(1,p=p):
                doChord = np.random.choice(np.arange(0,len(q),1),p= q)
                self.addSimpleChord(track, note, v,time)
            else:
                doChord -=1
                self.addNote(track, note, v, time)


    def twoNotes(self,track, notes, velosities, time):
        """
        not in use, add two notes "power chord"
        :param track:
        :param notes:
        :param velosities:
        :param time:
        :return:
        """
        track.append(Message('note_on', note=notes[0], velocity=velosities[0], time=0))
        track.append(Message('note_on', note=notes[1], velocity=velosities[1], time=0))

        track.append(Message('note_off', note=notes[0], velocity=velosities[0], time=time))
        track.append(Message('note_off', note=notes[1], velocity=velosities[1], time=0))

    def createSong(self,ticks_per_beat=480):
        """
        set a new song
        :param ticks_per_beat:
        :return:
        """
        self.mid = MidiFile(clip=True)
        self.mid.ticks_per_beat = ticks_per_beat

        track = MidiTrack()
        self.mid.tracks.append(track)
        tempo = 500000
        track.append(MetaMessage('set_tempo', tempo=tempo, time=0))


    def save_creation(self,fileName = "trying_hard"):
        """
        save the song as midi file
        :param fileName:
        :return:
        """
        print(f'saving file {fileName}.mid')
        self.mid.save(fileName + ".mid")


    def add_midi(self, backGround = " BasicSwing.mid", backgroundTimeCorection = 4, repatitions = 20):
        """
        add a background midi to the song
        :param backGround:
        :param backgroundTimeCorection:
        :param repatitions:
        :return:
        """
        if not backGround == None:
            drums = MidiFile(backGround)
            for j in range(len(drums.tracks)):
                for msg in drums.tracks[j]:
                    if msg.is_meta:
                        continue
                    msg.time *= backgroundTimeCorection  # int(msg.velocity/2.0)
            for i in range(len(drums.tracks)-1):
                track = MidiTrack()
                self.mid.tracks.append(track)
                for r in range(repatitions):
                    track.append(drums.tracks[i+1])

    # def trunck_Matricses(self, low, high):
    #
    #     for m in range(len(self.sl.matricies)):
    #         for do_ in range(m+1):
    #             indexes = np.arange((low - self.lowestNote)*(self.sl.notesNum**(m - do_)),(high - self.lowestNote)* (self.sl.notesNum ** (m - do_)),1)
    #             indexess = [indexes + ((self.highestNote -self.lowestNote)**(m - do_)) * p for p in range(int(len(self.sl.matricies[m])/
    #                                                                                                           (self.highestNote -self.lowestNote)**(1+m - do_)))]
    #             indexess = np.array(indexess).flatten()
    #             self.sl.matricies[m] = self.sl.matricies[m] [indexess]
    #             # for range_ in np.arange(0,len(self.sl.matricies[m]),(self.highestNote -self.lowestNote)**(m - do_)):
    #             #     temp= self.sl.matricies[m][range_ : range_ + range_ + (self.highestNote -self.lowestNote)**(m - do_)]
    #             #     temp1 = temp[(low - self.lowestNote)*(self.sl.notesNum**(m - do_)):(high - self.lowestNote)* (self.sl.notesNum ** (m - do_))]
    #             #     self.sl.matricies[m] = np.append(np.append(self.sl.matricies[m][:range_], temp1),self.sl.matricies[m][range_ + (self.highestNote -self.lowestNote)**(m - do_):])
    #
    #         self.sl.matricies[m] = self.sl.matricies[m].T[low - self.lowestNote: high - self.lowestNote].T
    #     self.sl.lowestNote = low
    #     self.sl.highestNote = high
    #     self.sl.notesNum = high - low +1
    #     print("this is how we do it")

    def next_rhythm(self, num):
        """
        get the next rhythm to play, a new one or the one fro the main channel
        :param num:
        :return:
        """
        if len(self.rhythms_played) <= num or len(self.rhythms_played)==0:
            rhythm, rhythmIndex = self.getRhythm(self.rhythms_played)
            return rhythm, rhythmIndex
        if self.rhythms_played[num] == -1: return self.narativeRhythm, -1
        return self.rhythms[self.rhythms_played[num]] ,self.rhythms_played[num]

    def add_spasipic(self, rhythm, vel,notes, repititions, instrument):
        """
        add spasific part of midi in new channel
        :param rhythm:
        :param vel:
        :param notes:
        :param repititions:
        :param instrument:
        :return:
        """
        track = MidiTrack()
        self.mid.tracks.append(track)
        self.mid.tracks[0].append(Message('program_change',channel = len(self.mid.tracks)-1, program=instrument, time=0))

        for i in range(repititions):
            self.addNotes(track,notes,rhythm,vel,self.mid.ticks_per_beat*4)

    def add_track_same(self, app_notes = 0, instrument = 57, pChords = 0, q=[1.0/8,1.0/4,1.0/4,1.0/8,1.0/8,1.0/16,1.0/16], timeLine=[]):
        """
        add the same thing(notes and rhythm, as the main channel, but can shift the octaves
        :param app_notes:
        :param instrument:
        :param pChords:
        :param q:
        :param timeLine:
        :return:
        """
        if len(timeLine) == 0: return
        if min(timeLine) <= 0 and len(self.savedNotes)==0: raise BaseException('first track must be full timed')
        track = MidiTrack()
        self.mid.tracks.append(track)
        self.mid.tracks[0].append(Message('program_change',channel = len(self.mid.tracks)-1, program=instrument, time=0))

        for i in range(len(self.rhythms_played)):
            # notes = self.savedNotes[i]
            #print(self.rhythms_played[i])
            if timeLine[i] == 0:
                self.addNotes(track,self.savedNotes[i],self.rhythms[self.rhythms_played[i]],np.zeros(len(self.savedNotes[i])),self.mid.ticks_per_beat*4, addition=app_notes, q = q, p=pChords)
                continue
            else:
                self.addNotes(track,self.savedNotes[i],self.rhythms[self.rhythms_played[i]],self.velocities[self.rhythms_played[i]],self.mid.ticks_per_beat*4, addition=app_notes, q = q, p=pChords)



    def add_track(self, startingNotes  = [48, 52, 55], app_notes = 0, returnRhytem=0.2, narativeP = 0., instrument = 57, pChords = 0, q=[1.0/8,1.0/4,1.0/4,1.0/8,1.0/8,1.0/16,1.0/16], timeLine=[]):
        """
        add the main track do this function only ones a song
        :param startingNotes:
        :param app_notes:
        :param returnRhytem:
        :param narativeP:
        :param instrument:
        :param pChords:
        :param q:
        :param timeLine:
        :return:
        """
        if(len(self.savedNotes) !=0): raise BaseException('use this function only ones in a-song')
        if len(timeLine) == 0: return
        if min(timeLine) <=0 and len(self.savedNotes)==0: raise BaseException('first track must be full timed')
        track = MidiTrack()
        self.mid.tracks.append(track)
        self.mid.tracks[0].append(Message('program_change',channel = len(self.mid.tracks)-1, program=instrument, time=0))
        self.sl.normalizeMatricies()
        notes = startingNotes
        rhythm, rhythmIndex = self.next_rhythm(0)
        self.rhythms_played.append(rhythmIndex)
        rhythm_played=True
        rhythmLen = len(rhythm)
        velocity = self.velocities[rhythmIndex]
        notes.extend(self.getNotes(startingNotes, velocity=velocity, numberOfNotes=rhythmLen - len(
            startingNotes), newArray = False))
        saveNotes=True
        self.savedNotes.append(notes)
        self.addNotes(track, notes, rhythm, velocity, self.mid.ticks_per_beat*4,p=pChords, q=q, addition=app_notes)

        i = 1
        while (i < len(timeLine)):
            if not np.random.binomial(1,returnRhytem) == 1 or rhythmIndex == -1:
                rhythm, rhythmIndex = self.next_rhythm(i)
            if not np.random.binomial(1, narativeP) == 0:
                rhythm = self.narativeRhythm
                rhythmIndex=-1
            self.rhythms_played.append(rhythmIndex)
            #print("rhythmIndex ",rhythmIndex)
            rhythmLen = len(rhythm)

            velocity = self.get_v(rhythmIndex)
            notes = self.getNotes(notes[-self.xNotes:], numberOfNotes=rhythmLen, velocity=velocity)
            self.savedNotes.append(notes)

            self.addNotes(track, notes, rhythm, velocity, self.mid.ticks_per_beat*4, p=pChords, q=q, addition=app_notes)
            i += 1


    def add_track2(self , startingNotes  = [48, 52, 55], app_notes = 0, instrument = 57, pChords = 0, q=[1.0/8,1.0/4,1.0/4,1.0/8,1.0/8,1.0/16,1.0/16], timeLine=[]):
        """
        add a track that use the same rhythm as the main channel but improvise the notes
        :param startingNotes:
        :param app_notes:
        :param instrument:
        :param pChords:
        :param q:
        :param timeLine:
        :return:
        """
        if len(timeLine) == 0: return
        if len(self.rhythms_played)==0: raise BaseException('first track must be full timed')
        track = MidiTrack()
        self.mid.tracks.append(track)
        self.mid.tracks[0].append(Message('program_change', channel = len(self.mid.tracks), program=instrument, time=0))

        self.sl.normalizeMatricies()
        notes = startingNotes
        rhythmIndex = self.rhythms_played[0]
        rhythm = self.get_r(rhythmIndex)
        rhythmLen = len(rhythm)
        velocity = self.velocities[rhythmIndex]
        notes.extend(self.getNotes(startingNotes, velocity=velocity, numberOfNotes=rhythmLen - len(
            startingNotes), newArray = False))
        if timeLine[0] == 0:
            velocity = np.zeros(len(velocity))
            notes = self.savedNotes[0]
        self.addNotes(track, notes, rhythm, velocity, self.mid.ticks_per_beat*4,p=pChords, q=q, addition=app_notes)

        i = 1
        while (i < len(timeLine)):
            rhythmIndex = self.rhythms_played[i]
            rhythm = self.get_r(rhythmIndex)
            #print(rhythmIndex)

            rhythmLen = len(rhythm)

            velocity = self.get_v(rhythmIndex)

            notes = self.getNotes(notes[-self.xNotes:], numberOfNotes=rhythmLen, velocity=velocity)

            if timeLine[i] == 0:
                velocity = np.zeros(len(velocity))
                notes = self.savedNotes[i]
            self.addNotes(track, notes, rhythm, velocity, self.mid.ticks_per_beat*4, p=pChords, q=q, addition=app_notes)
            i += 1
