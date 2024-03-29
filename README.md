# JohnChain
	Final project for the "67842 INTRODUCTION TO ARTIFICIAL INTELLIGENCE" course at HUJI (Spring 2019)
	See pdf for full details.


### Writers: 
	Ela Fallik, Liron Sade, Yoni Leibner, Shahaf Bassan



### Description:
    This project aims to generate a human-like jazz improvisation using existing jazz melodies.



### Files:
    runModel.py - Driver for our software.
    learn.py - Learns the notes
    Player.py - Interface for the notes and the rhythms.
    rhythemsData.py - Rhythm data. The rhythms we used in the model.
    params.py - Parameters for the model.
    JohnChain_TheOneAndOnly.mid - An output example
    midiToLearn - Folder of MIDI files to learn from.
    requirements.txt - Required python packages. install with ```pip install -r requirements.txt```
    Makefile
    README



### API:

	The setup of the model:
	Create Player objects, with the following params: 
		folder_name: the name of the folder where the data Midi files are.
		Rhythms: a list of rhythms. Note that the length of a rhythm needs to be the same as it's velocity list.
		Velocities: a list of velocities for the notes.
		xNotes: the order of the markov chain (number of notes back).
		lowestNote: the lowest note in the model.
		highestNote: the highest note in the model.
		scaleDict: dictionary with the Midi file names (from the folder folder_name) to learn from as keys, and their scale corrections (int) as values. Needed in order to set all data melodies to the same scale.
		narativeRhythm: the narrative rhythm, same length as narativeVelocity.
		narativeVelocity: the narrative velocity
		narrativeNotes: the narrative notes.
		rhythmDiffFunc: loss function for two rhythms (get the two rhythm indexes in the Rhythms list).
		uniformExplorationRateBetween: probability to do Uniform Exploration between rhythms.
		uniformExplorationRateWithin: probability to do Uniform Exploration inside a rhythm.
		closeExplorationRateBetwee: probability to do Close Exploration between rhythms.
		closeExplorationRateWithin: probability to do Close Exploration inside a rhythm.
		narrativeRateBetween: probability to do Narrative Exploration between rhythms.
		narrativeRateWithin:  probability to do Narrative Exploration inside a rhythm.
		scaleHeuristic: the notes, to change their learning reward,(list of lists).
		scaleHeuristicP: how much to change the chosen notes in scaleHeuristic, sould be in the same length.

	createSong:
	To start a song you need the method createSong, that takes:
		Ticks_per_beat: the number of ticks per beat in the song

	save_creation:
	To save the song you have the method: save_creation, it takes:
		melodyName: a string for the song name.

	Adding channels:
	add_track:
	Adding the first main channel: add_track (don't use it again in this song!), that takes:
		startingNotes: the notes to start from, need to be the same length as Xnotes in the learning process.
		App_notes: int, how many notes to shift the scale.
		returnRhytem: int [0,1] the probability to return the last rhythm.
		narativeP: int [0,1] the probability to do the narrative rhythm.
		Instrument: the instrument number (MIDI param).
		pChords: int [0,1] the probability to do a chord.
		q: Each index is the probability of a chord to repeat that amount of times. Should sum up to 1.
		timeLine: How many rhythms in the melody. (length of the melody)


	Add_track_same:
	Adding another channnel with the same notes as the main one, with different scale, and the same rhythms. takes:
		App_notes: int, how many notes to shift the scale.
		Instrument: the instrument number (MIDI param).
		pChords: int [0,1] the probability to do a chord.
		q: Each index is the probability of a chord to repeat that amount of times.
		    should sum up to 1.
		timeLine: Visibility for this channel. Same length as the main channel timeLine.


	Add_track2:
	Adding another channnel with different notes and the same rhythms. takes:
		startingNotes: the notes to start from, need to be the same length as xNotes in
				the learning process.
		App_notes:  int, how many notes to shift the scale.
		Instrument: the instrument number.
		pChords: int [0,1] the probability to do a chord.
		q: Each index is the probability of a chord to repeat that amount of times.
		    should sum up to 1.
		timeLine: Visibility for this channel. Same length as the main channel timeLine.
