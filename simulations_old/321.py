from functions_321 import main


shuffle_loc = 0

if __name__ == "__main__":
    shuffle_loc = 1     #1: after smoothing, 2: after binning, 3: after spike generation
    main(shuffle_loc)