#################################################
#
# this is the data we working on in the model
#
#################################################

from rhythemsData import *

# Herbie style- chamelon

# dictionarys of the songs you whant to learn, the numbers are the corection of the songs to C chord
dict1 = {"Ray Charles - Georgia On My Mind3": 0, "Take-Five3": -6,
         "new_song21": 2, "new_song22": 2,
         "new_song23": 2, "new_song24": 2, "new_song27": 2,
         "new_song29": 2, "C PARKER Yardbird suite": 0,
         "M DAVIS All blues1": -7, "Stevie Wonder - Higher Ground3": 9,
         "Stevie Wonder - Higher Ground6": 9,
         "Stevie Wonder - Boogie On Reggae Woman": 0,
         "dont_know_why-Norah-Jones-kar_ccm7": -10,
         "JohnColtrane_BlueTrain_FINAL": -3,
         "MilesDavis_SoWhat_FINAL": -2,
         "JohnColtrane_SoWhat_FINAL": -2,
         "JohnColtrane_Nutty_FINAL": 2,
         "HerbieHancock_Dolores_FINAL": 0,
         "JohnColtrane_BluesByFive_FINAL": 2,
         "PaulDesmond_BlueRondoALaTurk_FINAL": 7,
         "MilesDavis_BluesByFive_FINAL (1)": 2,
         "JohnColtrane_MyFavoriteThings-1_FINAL (1)": -7,
         "ChrisPotter_Rumples_FINAL": -7,
         "ChrisPotter_Anthropology_FINAL": 2,
         "MilesDavis_Eighty-One_FINAL": -5,
         "MilesDavis_K.C.Blues_FINAL": 0,
         "CannonballAdderley_SoWhat_FINAL": 0,
         "JohnColtrane_Oleo_FINAL": 2,
         "JohnColtrane_Soultrane_FINAL": -2,
         "JohnColtrane_Nutty_FINAL (1)": 2,
         "JohnColtrane_Countdown_FINAL": 2,
         "JohnColtrane_Trane'sBlues_FINAL (1)": 2,
         "MilesDavis_Oleo-1_FINAL": 2,
         "JohnColtrane_Mr.P.C._FINAL": -4,
         "CharlieParker_Ko-Ko_FINAL": 2,
         "JohnnyDodds_GotNoBlues_FINAL": -5,
         "JohnnyDodds_HeebieJeebies_FINAL": 4,
         "KidOry_GotNoBlues_FINAL": -5,
         "JohnnyDodds_HotterThanThat_FINAL": -3,
         "JohnnyDodds_MuskratRamble_FINAL": 4,
         "JohnnyDodds_OnceInAWhile_FINAL": -5,
         "KidOry_GutBucketBlues_FINAL": 0,
         "KidOry_SavoyBlues_FINAL": 4,
         "DickieWells_Dickie'sDream_FINAL": -1,
         "KidOry_Who'sIt_FINAL": 0,
         "KidOry_MuskratRamble_FINAL": 4,
         "DexterGordon_Cheesecake_FINAL": -3,
         "DexterGordon_SocietyRed_FINAL": 2,
         "DexterGordon_StanleyTheSteamer_FINAL": 2,
         "BuckClayton_AfterTheatreJump_FINAL (1)": -1,
         "BuckClayton_Dickie'sDream_FINAL": -3,
         "BuckClayton_DestinationK.C._FINAL": 0,
         "CharlieParker_EmbraceableYou_FINAL": -5,
         "LouisArmstrong_BigButterAndEggMan_FINAL": 4,
         "CharlieParker_HowDeepIsTheOcean_FINAL": -3,
         "LouisArmstrong_BasinStreetBlues_FINAL": 2,
         "LouisArmstrong_CornetChopSuey_FINAL": -5,
         "LouisArmstrong_OnceInAWhile_FINAL": -5,
         "LouisArmstrong_GutBucketBlues_FINAL": 0,
         "LouisArmstrong_MuskratRamble_FINAL": 4,
         "LouisArmstrong_SavoyBlues_FINAL": 5,
         "LouisArmstrong_GotNoBlues_FINAL": -5,
         "LesterYoung_Dickie'sDream_FINAL": -3,
         "BobBerg_SecondSight_FINAL": 2,
         "LesterYoung_AfterTheatreJump_FINAL": -1,
         "BobBerg_NoMoe_FINAL": 2,
         "BobBerg_BluesForBela_FINAL": -3,
         "BuckClayton_AfterTheatreJump_FINAL": -1,
         "BennyGoodman_HandfulOfKeys_FINAL": -3,
         "BennyGoodman_Avalon_FINAL": -3,
         "BennyGoodman_Nobody'sSweetheart_FINAL": 4,
         "BennyGoodman_TigerRag-2_FINAL": 4,
         "BennyGoodman_Runnin'Wild_FINAL": 2,
         "BenWebster_MyIdeal_FINAL": 2,
         "BennyGoodman_TigerRag-1_FINAL": 4,
         "BenWebster_ByeByeBlackbird_FINAL": 5,
         "ArtPepper_Stardust-2_FINAL": -3,
         "ArtPepper_Stardust-1_FINAL": -3,
         "ArtPepper_InAMellowTone_FINAL": -3,
         "ArtPepper_Desafinado_FINAL": -5,
         "BennyGoodman_Whispering_FINAL": -3,
         "SteveColeman_Cross-Fade-1_FINAL": 0,
         "SteveColeman_Cross-Fade-2_FINAL": 0,
         "PepperAdams_HowHighTheMoon_FINAL": 4,
         "WoodyShaw_Rosewood_FINAL": 0,
         "PepperAdams_EarlyMorningMood_FINAL": 4,
         "SonnyRollins_Playin'InTheYard-2_FINAL": 4,
         "SonnyRollins_Playin'InTheYard-1_FINAL (1)": 4,
         "ChrisPotter_Togo_FINAL": 5,
         "KennyGarrett_BrotherHubbard-1_FINAL": -5,
         "JoshuaRedman_TearsInHeaven_FINAL": 2,
         "JoshuaRedman_IGotYou_FINAL": -2
         }

dictFunk = {"new_song21": 2, "new_song22": 2,
            "new_song23": 2, "new_song24": 2, "new_song27": 2,
            "new_song29": 2,
            "M DAVIS All blues1": -7, "Stevie Wonder - Higher Ground3": 9,
            "Stevie Wonder - Higher Ground6": 9,
            "Stevie Wonder - Boogie On Reggae Woman": 0,
            "HerbieHancock_Dolores_FINAL": 0,

            "SteveColeman_Cross-Fade-1_FINAL": 0,
            "SteveColeman_Cross-Fade-2_FINAL": 0,
            "PepperAdams_HowHighTheMoon_FINAL": 4,
            "WoodyShaw_Rosewood_FINAL": 0,
            "PepperAdams_EarlyMorningMood_FINAL": 4,
            "SonnyRollins_Playin'InTheYard-2_FINAL": 4,
            "SonnyRollins_Playin'InTheYard-1_FINAL (1)": 4,
            "ChrisPotter_Togo_FINAL": 5,
            "KennyGarrett_BrotherHubbard-1_FINAL": -5,
            "JoshuaRedman_TearsInHeaven_FINAL": 2,
            "JoshuaRedman_IGotYou_FINAL": -2
            }

dictJazz = {"Ray Charles - Georgia On My Mind3": 0, "Take-Five3": -6,
            "C PARKER Yardbird suite": 0,
            "dont_know_why-Norah-Jones-kar_ccm7": -10,
            "COLTRANE.Grand central5": 4,
            "COLTRANE.Grand central6": 4,
            "COLTRANE.Impressions10": 7,
            "COLTRANE.Impressions8": 7,
            "COLTRANE.Impressions9": 7,
            "JohnColtrane_BlueTrain_FINAL": -3,
            "MilesDavis_SoWhat_FINAL": -2,
            "JohnColtrane_SoWhat_FINAL": -2,
            "JohnColtrane_Nutty_FINAL": 2,
            "HerbieHancock_Dolores_FINAL": 0,
            "JohnColtrane_BluesByFive_FINAL": 2,
            "PaulDesmond_BlueRondoALaTurk_FINAL": 7,
            "MilesDavis_BluesByFive_FINAL (1)": 2,
            "JohnColtrane_MyFavoriteThings-1_FINAL (1)": -7,
            "ChrisPotter_Rumples_FINAL": -7,
            "ChrisPotter_Anthropology_FINAL": 2,
            "MilesDavis_Eighty-One_FINAL": -5,
            "MilesDavis_K.C.Blues_FINAL": 0,
            "CannonballAdderley_SoWhat_FINAL": 0,
            "JohnColtrane_Oleo_FINAL": 2,
            "JohnColtrane_Soultrane_FINAL": -2,
            "JohnColtrane_Nutty_FINAL (1)": 2,
            "JohnColtrane_Countdown_FINAL": 2,
            "JohnColtrane_Trane'sBlues_FINAL (1)": 2,
            "MilesDavis_Oleo-1_FINAL": 2,
            "JohnColtrane_Mr.P.C._FINAL": -4,
            "CharlieParker_Ko-Ko_FINAL": 2,

            "JohnnyDodds_GotNoBlues_FINAL": -5,
            "JohnnyDodds_HeebieJeebies_FINAL": 4,
            "KidOry_GotNoBlues_FINAL": -5,
            "JohnnyDodds_HotterThanThat_FINAL": -3,
            "JohnnyDodds_MuskratRamble_FINAL": 4,
            "JohnnyDodds_OnceInAWhile_FINAL": -5,
            "KidOry_GutBucketBlues_FINAL": 0,
            "KidOry_SavoyBlues_FINAL": 4,
            "DickieWells_Dickie'sDream_FINAL": -1,
            "KidOry_Who'sIt_FINAL": 0,
            "KidOry_MuskratRamble_FINAL": 4,
            "DexterGordon_Cheesecake_FINAL": -3,
            "DexterGordon_SocietyRed_FINAL": 2,
            "DexterGordon_StanleyTheSteamer_FINAL": 2,
            "BuckClayton_AfterTheatreJump_FINAL (1)": -1,
            "BuckClayton_Dickie'sDream_FINAL": -3,
            "BuckClayton_DestinationK.C._FINAL": 0,
            "CharlieParker_EmbraceableYou_FINAL": -5,
            "LouisArmstrong_BigButterAndEggMan_FINAL": 4,
            "CharlieParker_HowDeepIsTheOcean_FINAL": -3,
            "LouisArmstrong_BasinStreetBlues_FINAL": 2,
            "LouisArmstrong_CornetChopSuey_FINAL": -5,
            "LouisArmstrong_OnceInAWhile_FINAL": -5,
            "LouisArmstrong_GutBucketBlues_FINAL": 0,
            "LouisArmstrong_MuskratRamble_FINAL": 4,
            "LouisArmstrong_SavoyBlues_FINAL": 5,
            "LouisArmstrong_GotNoBlues_FINAL": -5,
            "LesterYoung_Dickie'sDream_FINAL": -3,
            "BobBerg_SecondSight_FINAL": 2,
            "LesterYoung_AfterTheatreJump_FINAL": -1,
            "BobBerg_NoMoe_FINAL": 2,
            "BobBerg_BluesForBela_FINAL": -3,
            "BuckClayton_AfterTheatreJump_FINAL": -1,
            "BennyGoodman_HandfulOfKeys_FINAL": -3,
            "BennyGoodman_Avalon_FINAL": -3,
            "BennyGoodman_Nobody'sSweetheart_FINAL": 4,
            "BennyGoodman_TigerRag-2_FINAL": 4,
            "BennyGoodman_Runnin'Wild_FINAL": 2,
            "BenWebster_MyIdeal_FINAL": 2,
            "BennyGoodman_TigerRag-1_FINAL": 4,
            "BenWebster_ByeByeBlackbird_FINAL": 5,
            "ArtPepper_Stardust-2_FINAL": -3,
            "ArtPepper_Stardust-1_FINAL": -3,
            "ArtPepper_InAMellowTone_FINAL": -3,
            "ArtPepper_Desafinado_FINAL": -5,
            "BennyGoodman_Whispering_FINAL": -3
            }

# # #funk you can set other rhythm for the jazz vs funk songs

narrativeNotes = [38, 39, 40], [64, 67], [71, 72], [70, 72]
# jazz
narativeRhythmJazz = [1.0 / 6, 1.0 / 12, 1.0 / 6, 1.0 / 12, 1.0 / 6, 1.0 / 12, 1.0 / 12, 1.0 / 12, 1.0 / 12, 1.0 / 6,
                  1.0 / 12, 1.0 / 6, 1.0 / 12, 1.0 / 6, 1.0 / 12, 1.0 / 12, 1.0 / 12, 1.0 / 12]

narativeVelocityJazz = [120, 50, 120, 50, 120, 50, 0, 120, 50, 120, 50, 120, 50, 120, 50, 0, 120, 50]

narativeRhythmFunk = [1.0 / 6, 1.0 / 12, 1.0 / 12, 1.0 / 12, 1.0 / 12, 1.0 / 4, 1.0 / 12, 1.0 / 12, 1.0 / 12, 1.0 / 6,
                  1.0 / 12, 1.0 / 12, 1.0 / 12, 1.0 / 12, 1.0 / 4, 1.0 / 12, 1.0 / 12, 1.0 / 12]
narativeVelocityFunk = [120, 0, 120, 0, 120, 0, 50, 70, 90, 120, 0, 120, 0, 120, 0, 50, 70, 90]
