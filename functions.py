import os, requests, sys, time, datetime


def br():
    print('\n')


def download(url: str, dest_folder: str, chunk_size: int):

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # Maak de map aan als die nog niet bestaat
    
    bestandsNaam = url.split('/')[-1].replace(" ", "_") # -1 om het laatste ding van de splits te krijgen,
    # het replacen zodat we geen spaties krijgen in de bestandsnaam
    bestandsPad = os.path.join(dest_folder, bestandsNaam)

    if os.path.exists(bestandsPad):

        print("Het bestand bestaat al! Er wordt geprobeerd het te overschrijven, mits dit niet lukt, moet u het zelf verwijderen:")
        print(os.path.abspath(bestandsPad))
        br()

    request = requests.get(url, stream = True)

    # Controleer of de request is geaccepteerd
    if request.ok:
        print("Bestand downloaden naar ", os.path.abspath(bestandsPad), " === dit kan even duren...")
        beginTijd = time.time()
        with open(bestandsPad, 'wb') as bestand:
            
            total_length = int(request.headers.get('content-length'))
            totaleChunks = round(total_length / chunk_size)
            
            dl = 0

            chunkNo = 0

            for chunk in request.iter_content(chunk_size = chunk_size):

                if chunk:
                    chunkNo += 1
                    
                    dl += len(chunk)

                    bestand.write(chunk)
                    bestand.flush()
                    os.fsync(bestand.fileno())


                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s] chunk %s van de %s chunks..." % ('=' * done, ' ' * (50 - done), str(chunkNo), str(totaleChunks)))
                    sys.stdout.flush()

        eindTijd = time.time()
        downloadDuur = datetime.timedelta(seconds=(eindTijd - beginTijd))
        print("Download is gelukt! Het duurde {}!".format(str(downloadDuur)))
        br()
    
    else:
        print("Download niet gelukt: status code {}\n{}".format(request.status_code, request.text))
        print("Probeer het bestand zelf te downloaden via {} en in de map {} te zetten!".format(url, dest_folder))