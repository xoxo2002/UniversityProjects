import random

def main():
    # initialisation
    code_breaker = []
    mastermind = []
    # choices of colour 
    colours = ['red', 'blue', 'green','purple','yellow','pink','white','orange']
    attempts = 0
    #initiliase as sentinel value
    hint1 = 0

    # instructions and welcome message
    print('''☆ ☆ ☆ ☆ WELCOME TO MASTERMIND! ☆ ☆ ☆ ☆

Instructions to play:
1) From the list of colours below, pick one colour.
2) Key your choice of colour at the prompt.
3) Repeat and keep in mind that your colours will be arranged in this sequence:
   [colour1]  [colour 2]  [colour 3]  [colour 4]
4) If you wish to quit the programme midway, simply enter 'quit'. ''')
    print('=================================================================================')
    
    # show user the color choices
    print('COLOURS :  ', end = '')
    for j in colours:
        print(j,'  ', end = '')
    print()
   
    # generate code_maker colours 
    mastermind = (generate_colours(colours))
    
    # generate code_breaker colours(input)
    while hint1 != 4:
        print()
        code_breaker = (guess_colours(colours))
        attempts += 1
        # compare mastermind to code_breaker lists
        hint1, hint2 = compare(mastermind, code_breaker)
        print(f"""--------------------------------------------
Correct colour in the correct place: {hint1}
Correct colour in the wrong place: {hint2}
--------------------------------------------""")

    #print congrats message
    print(f"""
°˖✧◝(⁰▿⁰)◜✧˖° CONGRATULATIONS, YOU'VE GUESSED IT! °˖✧◝(⁰▿⁰)◜✧˖°
ATTEMPT(S): {attempts}
=================================================================================""")

#mastermind
def generate_colours(list):
    choice = []
    for y in range(4):
        choice.append(random.choice(list))
    return choice

#code_breaker
def guess_colours(list):
    # initialise list for user input
    guess = []
    # prompt user for 4 colours 
    for x in range(4):
        pick = input('colour {}: '.format(x+1)).strip().lower()
            
        valid = validate(list, pick)
        #if input not valid reprompt user for colour
        while valid == 0:
            pick = input('colour {}: '.format(x+1)).strip().lower()
            valid = validate(list, pick)
        #if input valid, append to the list
        if valid == 1:
            guess.append(pick)
            
    print('Your guess is:', guess)
    return guess

#error checking code 
def validate(list, pick):
    valid = 0
    if pick == 'quit':
            quit = input('Do you wish to quit the program? (yes/no)')
            if quit == 'yes':
                print("Alright, till' next time! Goodbye.")
                exit()
            else:
                print('Let us continue!')
                
    for i in list:
        if(pick == i):
            valid += 1
    #if not valid print error message
    if valid == 0:
        print('Select colours in the in choice list only. Pick again')
        valid = 0
        
    return valid

#compare mastermind and code_breaker
def compare(maker, breaker):
    #initialise variables 
    placeandcolour = 0
    colour = 0
    m = 0
    n = 0
    copy1 = maker.copy()
    copy2 = breaker.copy()
 
    #correct colour correct place
    for i in range(4):
        for j in range(4):
            if maker[i] == breaker[j]:
                if i == j:
                    copy1[i] = '0'
                    copy2[j] = '0'
            
    #correct colour wrong place
    for m in range(4):
        for n in range(4):
            if(copy1[m] == copy2[n]):
                if(copy1[m] != '0'):
                    copy1[m] = '1'
                    copy2[n] = '1'
                   
    #note the number of 1 and 0
    for a in range(4):
        if(copy2[a] == '0'):
            placeandcolour += 1
        if(copy2[a] == '1'):
            colour += 1
                
    return placeandcolour, colour
    


if __name__ == '__main__':
    main()
