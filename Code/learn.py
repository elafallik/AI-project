##############################################################################
#
# the markov model for the notes
#
##############################################################################

import numpy as np
import mido
from mido import Message, MidiFile, MidiTrack
from copy import copy

RANDOM_NOTE_SKIPS = [12, 7, 4, -5, -8, -12]

class SongLearner:
    def __init__(self, lowestNote=0, highestNote=127, notesBack=2, narrativeNotes=None,
                 additionVec = None):

        assert lowestNote >= 0 and highestNote <= 127 and lowestNote <= highestNote

        self.narrativeNotes = narrativeNotes
        self.forcedNotes = []                   #todo fix this
        self.notesNum = highestNote - lowestNote + 1
        self.highestNote = highestNote
        self.lowestNote = lowestNote
        self.matricies = []
        for i in range(notesBack):
            self.matricies.append(np.zeros((self.notesNum ** (i + 1), self.notesNum)))
        self.notesBack = notesBack
        if not len(additionVec) == highestNote - lowestNote + 1:
            self.additionVec = np.ones(highestNote - lowestNote + 1)
        else:
            self.additionVec = additionVec


    def updateWeight(self, notes):
        matriciesToUpdate = min(len(notes) - 1, len(self.matricies))
        for i in range(matriciesToUpdate):
            relevantNotes = notes[-2 - i: -1]
            targetIdx = self.getMatrixIndex(relevantNotes)
            self.matricies[i][targetIdx, notes[-1] - self.lowestNote] += self.additionVec[notes[-1]-self.lowestNote]

    def normalizeMatricies(self):
        self.normalizedMatricies = []
        for m in self.matricies:
            x = np.sum(m, axis=1)[:, np.newaxis]
            x[np.argwhere(x == 0)] = np.inf
            self.normalizedMatricies.append(m / x)

    def learn(self, midiName, channel=None, noteShift=0):
        """
        :midiName path to MIDI file
        :channel which channel to learn
        """
        mid = mido.MidiFile(midiName)
        state = []
        for msg in mid:
            if msg.is_meta:
                continue
            if channel is not None and msg.channel != channel:
                continue
            if msg.type == 'note_on' and msg.velocity > 0:
                note = self.adjustNote(msg.note, noteShift)
                if note >= self.highestNote or note <= self.lowestNote:
                    state = []
                    continue
                if len(state) <= self.notesBack:
                    state.append(note)
                    if len(state) == self.notesBack + 1:
                        self.updateWeight(state)
                    continue
                state.pop(0)
                state.append(note)
                self.updateWeight(state)

    def learnSong(self, midiName, channels, noteShift):
        for channel in channels:
            self.learn(midiName, channel, noteShift)


    def nextNote(self, notes, uniformExplorationRate=0.20,
            closeExplorationRate=0.05, narrativeRate=0.02):
        """
        :notes k last notes
        :uniformExplorationRate rate at which to pick next note
            based off uniform distribution instead of the matricies
        :closeExplorationRate rate at which to pick next note
            close to last note
        """
        assert self.normalizedMatricies is not None
        assert uniformExplorationRate + closeExplorationRate + narrativeRate <= 1
        if self.forcedNotes:
            return self.forcedNotes.pop(0)
        UNIFORM, CLOSE, NARRATIVE, NORMAL = range(4)
        normalRate = 1 - uniformExplorationRate - closeExplorationRate - narrativeRate
        probabilities = [uniformExplorationRate, closeExplorationRate,
                narrativeRate, normalRate]
        path = np.random.choice(4, p=probabilities)
        if path == UNIFORM:
            return self.uniformNextNote(notes)
        elif path == CLOSE:
            return self.randomCloseNote(notes[-1])
        elif path == NARRATIVE:
            return self.narrativeNextNote()
        elif path == NORMAL:
            return self.normalNextNote(notes)

    def narrativeNextNote(self):
        assert self.narrativeNotes is not None
        self.forcedNotes = copy(self.narrativeNotes[np.random.choice(len(self.narrativeNotes))])
        return self.forcedNotes.pop(0)


    def normalNextNote(self, notes):
        rnd = np.random.rand()
        matriciesToCheck = min(len(notes), len(self.matricies))
        for i in reversed(range(matriciesToCheck)):
            relevantNotes = notes[-i - 1:]
            targetIdx = self.getMatrixIndex(relevantNotes)
            relevantMatrix = self.normalizedMatricies[i][targetIdx]
            sums = np.cumsum(relevantMatrix)
            possibleNotes = np.argwhere(sums >= rnd)
            if len(possibleNotes) > 0:
                return np.argmax(sums >= rnd) + self.lowestNote
        # No matches
        return self.randomCloseNote(notes[-1])


    def uniformNextNote(self, notes):
        # Uniform distribution
        matriciesToCheck = min(len(notes), len(self.matricies))
        for i in reversed(range(matriciesToCheck)):
            relevantNotes = notes[-i - 1:]
            targetIdx = self.getMatrixIndex(relevantNotes)
            relevantMatrix = self.normalizedMatricies[i][targetIdx]
            possibleNotes = np.argwhere(relevantMatrix > 0)
            if len(possibleNotes) == 1:
                continue
            if len(possibleNotes) > 0:
                return np.random.choice(possibleNotes.flatten()) + self.lowestNote
        return self.randomCloseNote(notes[-1])

    def randomCloseNote(self, note):
        new_note = note + np.random.choice(RANDOM_NOTE_SKIPS)
        return self.inRangeNote(new_note)

    def getMatrixIndex(self, notes):
        targetIdx = 0
        for note in notes:
            targetIdx *= (self.notesNum)
            targetIdx += (note - self.lowestNote)
        return targetIdx

    def inRangeNote(self, note):
        if note > self.highestNote:
            return self.highestNote
        if note < self.lowestNote:
            return self.lowestNote
        return note


    def adjustNote(self, note, shift):
        note = note + shift
        return self.inRangeNote(note)


def createSong(sl):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(Message('program_change', program=12, time=0))
    n1 = 36
    n2 = 38
    track.append(Message('note_on', note=n1, velocity=84, time=200))
    track.append(Message('note_off', note=n1, velocity=84, time=200))

    track.append(Message('note_on', note=n2, velocity=64, time=100))
    track.append(Message('note_off', note=n2, velocity=64, time=200))
    i = 0
    while (i < 100):
        n = sl.nextNote([n1, n2], 0.2)
        track.append(Message('note_on', note=n, velocity=84, time=200))
        track.append(Message('note_off', note=n, velocity=84, time=200))
        n1 = n2
        n2 = n
        i += 1
        n = sl.nextNote([n1, n2], 0.2)
        track.append(Message('note_on', note=n, velocity=64, time=100))
        track.append(Message('note_off', note=n, velocity=64, time=200))
        n1 = n2
        n2 = n
        i += 1
        if i % 30 == 0:
            n1 = 36
            n2 = 38 + 12


    mid.save('new_song1.mid')

if __name__ == '__main__':
    sl = SongLearner(128, 2)
    sl.learn('data/new_song21.mid', 1)
    sl.learn('data/new_song22.mid', 2)
    sl.learn('data/new_song23.mid', 3)
    sl.learn('data/new_song24.mid', 4)
    sl.learn('data/new_song27.mid', 7)
    sl.learn('data/new_song29.mid', 9)
    sl.normalizeMatricies()
    createSong(sl)


