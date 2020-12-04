# imports


# Functie om te vragen of dit de eerste keer is dat het programma wordt opgestart, voor nu nog niet echt gebruikt!

# bool recursed, String initString
def initProgramFirstCheck():
    checked = False
    initBool = False # Standaard nog niet geopend

    while not checked:

        initString = input("Is dit de eerste keer dat je het yolov3 programma opent? (Y/N)")
            
        if initString == "Y":
            initBool = True

        elif initString == "N":
            initBool = False
                
        else:
            print("Verkeerde input, start opnieuw...")
            continue
        
        checked = True
        
    return initBool


def __main__():
    antwoord = initProgramFirstCheck()
    print(antwoord)