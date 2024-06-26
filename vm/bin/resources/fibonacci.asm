# (-0 indicates unused value)

# initialize register 0 as n = 9
STOREW 9 0 0 0 1
# output will be stored in register 1

# initialize register 2 as 1, will be used as index
STOREW 1 0 2 0 1


# initialize register 3 as 1 (utility)
STOREW 1 0 3 0 1


# initialize [0]_2 as 0 and [1]_2 as 1 (initial values)
STOREW 0 0 0 0 2



STOREW 1 0 1 0 2


# once index = n, terminate (pc += 7)
BEQ 2 0 7 1 1


# increment index
FADD 2 2 3 1 1


# load [index - 2]_2, [index - 1]_2 into registers 4, 5
LOADW 4 -2 2 1 2



LOADW 5 -1 2 1 2


# calculate new value in register 6
FADD 6 4 5 1 1


# store new value in [index]_2
STOREW 6 0 2 1 2


# jump to beginning of loop
JAL 7 -6 -0 1 -0


# terminate
TERMINATE -0 -0 -0 -0 -0