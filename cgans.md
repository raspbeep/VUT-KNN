`ssh xkrato61@zenith.cerit-sc.cz`
qsub
qstat
qdel

pouzivat scratch local na vygenerovane data
-I interactive

`qsub -I -l select=1:ncpus=2:mem=4gb:scratch_local=1gb -l walltime=2:00:00`

get job ID po submite alebo v interactive jobe
`echo $PBS_JOBID`

`qstat -u xkrato61 @zenith.cerit-sc.cz`

gpu job
`qsub -I -q gpu -l select=1:ncpus=1:ngpus=1:scratch_local=10gb -l walltime=24:0:0`



*Terms*

**feature map**
	- vystup filtru a predoslej vrstvy
	- vseobecne su to aktivacie vrstvy v danej hlbke
	- v CNN kazde jadro detekuje iste features na vstupe a tie ponechava (zvyraznuje)
		- FM popisuje priestorove usporiadanie tychto features po aplikacii filtru

**instance normalization**
	- normalizuje features pre kazdu instance samostatne narozdiel od batch normalization, ktore norm. cez celu batch
	- pouzivana pri staly transferoch, zachovava nezavislost kazdeho obrazku
	- viac odolny pristup ku zmenam osvetlenia jednotlivych obrazkov v batchi
	- mechanism:
		1. Vypocitaj mean and variance feature mapy kazdej instancie(obrazku)
		2. Normalizuj (subtract mean and divide by sqrt(variance))
		3. Scale and shift using learnable parameters unique to each feature channel (γ and β, respectively)

**residual blocks**
	- vylepsuje schopnost ucit sa, odstranuje vanishing gradient problem
	- doplnene do stredu siete, pretoze apparently je lahsie doplnit extra bloky so shortcutami na odstranenie chyby predoslych blokov, nez propagovat chybu GD do hlbokych vrstiev (diminishing gradient)

**Conv2d vs. ConvTranspose2d**
	***Conv2d***
        - zmensuje dimenzie vstupu a zvysuje dimenzie features

	***ConvTranspose2d***
	   	- zvysuje dimenzie vystupu a zmensuje features
		- dilation (upscaling)- vlozi nuly medzi hodnoty input feature maps
		- aplikuje konvoluciu, interpoluje aby zvacsila obrazok
