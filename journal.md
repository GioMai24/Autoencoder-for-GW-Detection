## Gigi, 27/11/2025

installato miniconda x86 dal .sh scaricabile da anaconda.com/download
aggiunto recovery account: user: <recovery>, psw <quello che dici quando esplode la vm>
aggiunto recovery ai sudoers
scaricati i dataset in /mnt, estratti in AnomalyDetectionPaper, messo readme per distiguere i files.

## Gio, 28/11/2025

Spostato miniconda in /mnt.
Aliasato '..', '...' e 'tomnt' per cdare più velocemente ihih.
Ho creato un conda env di nome temp con dentro jupyter e h5py per leggere i file, sta in gioprova.

## Gigi, 28/11/2025

trovato datasets, scaricati e messi in mnt: H1.h5 e L1.h5


## Gio, 29/11/2025
Continuato a giocare con conda, ho installato temp tramite temp.yml in /mnt/gioprova, puoi vedere tutti i pacchetti installati. torch è disponibile solo tramite pip, pip e conda cozzano un po' insieme, se vuoi installare altri pacchetti tramite conda bisogna ricostruire l'environment e installare i pacchetti di pip per ultimi (guarda docs), sono di fretta, ciao.

## Gigi, 30/11/2025

Cercato di capire come sono i dati tipo. Cosa rappresentano e shit like that. 
Provato a visualizzare injection e noise (mi sembrano identici ma d'altronde non sono un AutoEncoder).
Ho scritto quello che ho trovato/capito di nuovo in un readme dentro la cartella data. 
Cosa dobbiamo capire però è: i dati che abbiamo non sono TimeSeries nel senso stretto. Molto facilmente convertibili in quel formato eh. Ma a noi interessa? Non so.
Cominciato a pensare a fare la classe dataset di pytorch. in /mnt/luigiprova

## gigi, 1/12/2025

Cercato di fare dataset classe di pytorch. Fatta di per se. Ma mi triggera con in multiprocess (num workers > 0) ottengo performances sensibilmente peggiori rispetto al seriale. Scaricato h5torch but non funziona out of the box sembra. Si potrebbe valutare di decomprimere i dati / idk. 
File utile è h5dataset.py

## gigi, 2 /12/2025

Rechunckato i dati sia in chunck di 4 righe (*1_rechuncked.h5) sia senza chunk, tutto contiguous (*1contiguous.h5).

## Gio, 2/12/2025

Creato ambiente conda `plswrk` con i tool teoricamente necessari a buildare hdf5 parallel. Ho scaricato [CMakeUserPresets.json](https://support.hdfgroup.org/documentation/hdf5/latest/cmake-presets.html#subsec_cmake_presets_files_json_details) che dovrebbe contenere le opzioni per buildare con cmake e enablare parallelizzazione.

Installato da sorgente MPI nell'ambiente conda. È stato veloce e indolore, non ho praticamente toccato le config flags, ma installando le tre versioni di hdf5 parallel i test andavano bene.

Devo ancora buildare hdf5, ho scaricato la versione 2.0.0 che è la più recente e dicono sia la più efficiente. Tuttavia hanno tolto gli autotools e il sistema di buildare con `./config`, ora bisogna usare `cmake` e guardarsi sti presets. Probabilmente è più facile di quel che sembra, ma i loro README non mi piacciono.

## Gio, TeCNiCAmenTE 3/12/2025

Sono riuscito ad installare (correttamente?) HDF5 nell'ambiente conda presumibilmente ha la flag della parallelizzazione. Ho installato h5py seguendo sta guida del [HDFGroup](https://www.hdfgroup.org/2025/07/22/how-to-build-hdf5-library-and-h5py-in-a-conda-virtual-environment-update/), **NOTA**: Dice che se HDF5 è installato a livello di sistema oltre che nell'env, potrebbe essere prima trovato quello di sistema... Che Dio ce la mandi buona, altrimenti piallo la vm e la rifaccio da capo.

L'ambiente `plswrk` contiene attualmente solo h5py praticamente, bisogna installare ancora jupyter e tutto il resto. Dato che HDF5 e MPI sono stati installati manualmente ogni buona norma di installazione è stata violentata: sono stati mischiati conda, installazioni da sorgente e pip. Meno male che hanno inventato HDF5, così so verso chi rivolgere il mio odio.


## Gio, still 3/12/2025 but morning

La parallelizzazione funziona, ho cambiato qualche riga in `gridrunner` e `h5dataset`, le ho segnate con i commenti.

Ho creato un nuovo conda env `killgg` per testare il pacchetto senza HDF5 buildato manualmente. Funziona. Cosa abbiamo imparato da questa esperienza?
1. I worker di PyTorch leggono un batch a testa. Per cui la lettura di un singolo batch è più veloce con un solo processo, ma se si vogliono utilizzare più batch anca no.
2. Come si compila un programma da sorgente. Può essere divertente se fatto con calma e leggendo tutti i funny README. Non che io l'abbia fatto bene e con calma.
3. Avere un atteggiamento positivo. È fondamentale per mantenere un ambiente di lavoro efficiente e positivo.

## gigi & gio <3, 3/12 

LSTM recurrent network? idk sto cercando di copiare la architettura di moreno per capire un po'. Non capisco una sega da pytorch docs. Fixed dataset class e dataloader perchè traineremo AE solo con noise. TO DO: fix git shit. Fixed git shit. Capito un po' meno come ha fatto l'arkitettura il king eric moreno. letto lstm fino a sboccare.

## Gio, 4/12

Forse ho capito come funziona un nn.LSTM dovrebbe essere un'intera rete, per utilizzare un solo strato si può usare nn.LSTMCell (do not). Siccome ha un output strano ed è già una sequenza di layer, non si può mettere in nn.Sequential. In model.ipynb sul mio branch ci sono gli appunti sulla mia ricerca. Come al solito mi metto a scrivere il journal quando ho 30 secondi rimasti. Addio.

## Gio, 7/12

Aggiunto il repeating vector al model.ipynb in branch gio (secondo me dovremmo dare i nomi intelligenti ai branch). Ho fatto un po' di magheggi per spostare l'asse dei batch nei tensori e metterlo al primo posto, ma comunque l'output degli LSTM ha le dimensioni tutte storte e brutte, quindi penso che dipenderà da come imposteremo il tutto alla fine. Ora sto investigando l'ultimo layer del paper che è il TimeDistributed, non è implementato nativamente in PyTorch, penso proprio che dovremo fare una cosa custom. Da quello che ho capito è una collezione di layer densi: ognuno dei 32 canali di output del LSTM decoder ha una "rete neurale" con i 100 valori delle serie (un solo layer di sta roba). Potrebbe essere viceversa e potrei sbagliarmi.
