import os, requests, sys

def br():
    print('\n')

def download(url: str, dest_folder: str):

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # Maak de map aan als die nog niet bestaat
    
    bestandsNaam = url.split('/')[-1].replace(" ", "_") # -1 om het laatste ding van de splits te krijgen,
    # het replacen zodat we geen spaties krijgen in de bestandsnaam
    bestandsPad = os.path.join(dest_folder, bestandsNaam)

    request = requests.get(url, stream = True)

    # Controleer of de request is geaccepteerd
    if request.ok:
        print("Bestand downloaden naar ", os.path.abspath(bestandsPad))

        with open(bestandsPad, 'wb') as bestand:
            
            total_length = int(request.headers.get('content-length'))
            dl = 0

            for chunk in request.iter_content(chunk_size = 1024 * 8):

                if chunk:
                    
                    dl += len(chunk)

                    bestand.write(chunk)
                    bestand.flush()
                    os.fsync(bestand.fileno())


                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                    sys.stdout.flush()

        print("Download is gelukt: status code {}\n{}".format(request.status_code, request.text))
    
    else:
        print("Download niet gelukt: status code {}\n{}".format(request.status_code, request.text))
        print("Probeer het bestand zelf te downloaden via {} en in de map {} te zetten!".format(url, dest_folder))