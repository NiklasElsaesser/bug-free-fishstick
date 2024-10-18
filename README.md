# Assignment im Kurs Machine Learning Fundamentals

### Projekt Teilnehmer
* Anna Stöhrer da Silva
* Bernice Fabich
* Jan Schneeberg
* Niklas Elsässer

### Dozent
* Ruben Nuredini

## Setup
The Main work is seperated and developed in the following Notebooks:
1. [Simple LDA](https://github.com/NiklasElsaesser/bug-free-fishstick/blob/main/Simple_LDA.ipynb)
2. [Improved LDA](https://github.com/NiklasElsaesser/bug-free-fishstick/blob/main/Improved_LDA.ipynb)
3. [Benchmarks & Visualizations](https://github.com/NiklasElsaesser/bug-free-fishstick/blob/main/Benchmarks%26Visualizations.ipynb)

Just Clone the Repo and run the Notebooks, thats it.


# Inhaltsverzeichnis
- [Theoretische Grundlagen](#theoretische-grundlagen)
- [Simple LDA](https://github.com/NiklasElsaesser/bug-free-fishstick/blob/main/Simple_LDA.ipynb) (Separates Notebook)
- [Improved LDA](https://github.com/NiklasElsaesser/bug-free-fishstick/blob/main/Improved_LDA.ipynb) (Separates Notebook)
- [Benchmarks & Visualizations](https://github.com/NiklasElsaesser/bug-free-fishstick/blob/main/Benchmarks%26Visualizations.ipynb) (Separates Notebook)
- [Literaturverzeichnis](#literaturverzeichnis)


# Theoretische Grundlagen
Durch die Digitalisierung steigt die Anzahl digitaler Texte drastisch an, sodass diese kaum noch manuell zu erfassen sind. So beheimatet zum Beispiel das Deutsche Textarchiv (digitale Textsammlung) um die 145 Millionen Texte.
Diese Masse an Texten muss verwaltet werden, um Texte zu finden und abzubilden. Des Weiteren müssen sie von Computern ausgewertet werden können, da dies manuell nicht machbar ist.
Im Jahr 2007 beschäftigte sich Franco Moretti mit seinem Forschungsprogramm damit, tausend Werke der Weltliteratur gleichzeitig miteinander zu vergleichen. Möglich war dies aber nur mit (text-) statischen Verfahren und computergestützten Visualisierungen. (Heyer, Wiedemann, Niekler 2020, S.351-353)

## Topic Modeling
Topic Modelling ist eine bekannte Klasse dieser statischen Verfahren. Hierbei können große Daten-, Textmengen automatisch in Themenbereiche aufgeteilt werden. Es wird davon ausgegangen, dass jede Wortform zu einem Themenbereich (Topic) gehört. Durch die Untersuchung der Verteilung von zusammengehörigen Wortformen in einem Text sollen die Themenbereiche dieses Textes abgeleitet werden können.
Hiermit ist das Klassifizieren und Zusammenfassen von Texten und deren Abschnitten möglich. (Heyer, Wiedemann, Niekler 2020, S.351-353)

## Stärken und Schwächen
Stärken sind demnach die Unterteilung von großen Texten in Themen (Topics), sowie die Tatsache, dass dies voll automatisch geschieht.
Nachteil ist, dass Syntax-Strukturen, Regelmäßigkeiten der Aussagen und weitere Informationen nicht aus dem Topic Modelling resultieren.
Dennoch kann Topic Modelling als Vorarbeit für die weitergehenden Untersuchungen genutzt werden. (Heyer, Wiedemann, Niekler 2020, S.351-353)

### Step 3: Hilfsmittel für das LDA Model erstellen
Um das LDA Model später geordnet zu erzeugen, erstellen wir zwei Hilfsmittel.
### Dictionary: ID zu Wort
Corpora weißt jeden Wort eine eindeutige ID zu, mit welcher es später wieder erkannt werden kann.
<br>Dies hilft bei der schnellen und effizienten Verarbeitung der Wörter

### Liste: Korpus
Der Korpus listet für alle Dokumente einen Vektor welchen wir als "Bag of Words" bezeichnen.
<br>Er enthält alle Wörter und ihre Häufigkeit in dem Dokument. Über diesen Vektor, kann dem Dokument eine Bedeutung zugewiesen werden.

## LDA Modell
### Step 4: Das LDA Model trainieren
Das Latent Dirichlet Allocation (Latente Dirichlet Zuordnung (LDA)) Verfahren ist ein 3-stufiges Bayesian Model, um Topic Modeling durchzuführen (Blei et al.(2003)). Das Verfahren besteht aus den Teilen:
* **Latent:** Die Themen des Korpuses sind verborgen und müssen aus den Dokumenten und deren Wortverteilung abgeleitet werden (Blei et al. (2003), S. 1007).
* **Dirichlet:** Eine Verteilung, basierend auf dem Bayesian Modell, welche die Prioriverteilung (Anfangswahrscheinlichkeit) für die Themenverteilung in den Dokumenten darstellt und die Parameter beeinflussen wie stark bestimmte Themen in einem Dokument vertreten sind (Blei et al. (2003), S. 1007).
* **Allocation:** Die Prozess der Zuweisung und Verteilung beschreibt wie Wörter den verschiedenen Themen zugeordnet werden und wie die Themen in einem Dokument verteilt sind. Allocation beschreibt wie das Modell Wörter in einem Dokument aufgrund ihrer Wahrscheinlichkeit verschiedenen latenten Themen zugeordnet sind. Die Themen - Wörter verteilung in einem Dokument bildet die Grundlage der Analyse (Blei et al. (2003), S. 1007).

LDA nimmt an das es eine Themenverteilung für jeden Text gibt, welche berechnet wird indem ein Thema aus allen verfügbaren Themen $T={T_1,…,T_K}$ jedem Token ($≈$ Wort) eines Textes zugeordnet wird. Der Nutzer definiert dabei die Anzahl der modellierten Themen $K∈N$. Der Text (oder das Dokument) wird als Corpus bezeichnet und besteht aus Texten $M$ sowie Themen $T$ (Rieger at al. (2020 Juni), S. 120); Blei et al. (2003), S. 997).
Dabei ist $N^m$ die Größe des Textes und $W={W,…,W_V}$ die Menge der Wörter wobei $V∈N$ die Größe des implizierten Vokabulars (die Menge aller eindeutigen Wörter) ist(Rieger et al. (2020 Juni), S. 120).

<img src="visuals/page1.png" alt="p1" width="250" />
<br>
<img src="visuals/page2.png" alt="p2" width="250" />

Somit besteht ein $Dokument$ aus:
$$
D^{(m)} = (W_1^{(m)}, \dots, W_{N^{(m)}}^{(m)}), \quad W_n^{(m)} \in W; \quad n = 1, \dots, N^m
$$
Die Themenzuweisung für einen Text $m$ ist wie folgt:
$$
T^{(m)} = (T^{(m)}, \dots, T_{N^{(m)}}^{(m)}), \quad T_n^{(m)} \in T
$$
Jede Themenzuweisung $T_n^{(m)}$ hängt mit einem Token $W_n^{(m)}$ aus dem Text $m$ zusammen.

![page3.png](visuals/page3.png)

Um das Wahrscheinlichkeitsmodell für LDA aufzustellen sind folgende Definitionen und Annahmen notwendig: $n_k^{(mv)};k=1,…,K;v=1,…,V$ als Anzahl zugewiesener Wörter $v$ in Text $m$ zugehörig zu Thema $k$, dadurch lässt sich die Summe der Wörter $v$  in Thema $k$ über alle Dokumente $D$ mit $n_k^{(*v)}$ bestimmen. Wenn $w_k$ die Vektoren der Wortanzahl für $k=1,…,K$ Themen ist, dann lässt sich mit diesen Definitionen das Modell wie folgt aufstellen:

$$
W_n^{(m)} \mid T_n^{(m)}, \phi_k \sim \text{Discrete}(\phi_k), \quad \phi_k \sim \text{Dirichlet}(\eta)
$$

$$
T_n^{(m)} \mid \theta_m \sim \text{Discrete}(\theta_m), \quad \theta_m \sim \text{Dirichlet}(\alpha)
$$


Die Dirichlet Verteilungs Hyperparameter $α$ und $η$ müssen vom Nutzer deklariert werden. Da normalerweise keine a-priori (Anfangswahrscheinlichkeit) Informationen der Themen $θ$  und Wortverteilungen $ϕ$ vorliegen, werden $α$ und $η$ symmetrisch bestimmt (Rieger at al. (2020 Juni), S. 120).

Ein hoher $η$ Wert führt zu einem Verlust der Gleichmäßigkeit der Wortmischung pro Thema, ein niedriger $η$ Wert erhöht und verbessert die Gleichmäßigkeit wodurch weniger dominantere Wörter pro Thema zugeordnet werden. Nach dem gleichen Prinzip steuert α die Mischung der Themen in den Texten (Rieger at al. (2020 Juni), S. 120).

LDA ist eine Weiterentwicklung von LSI, bzw. pLSI, wobei LSI selbst eine Weiterentwicklung von tf-idf ist (Blei et al. (2003), S. 994). LSI verbessert tf-idf indem eine größere reduktion der Beschreibungslänge in großen Daten möglich ist und die statistische Struktur zwischen oder innerhalb von Dokumenten offenbart (Blei et al. (2003), S. 994; Rosario, B. (2000)). Probability LSI (pLSI) verbessert LSI dahingehend, dass alle Worte in einem Dokument eine Stichprobe aus einem Mischungsmodell sind in welchem wiederum verschiedene Themen Multinomial-Verteilungen abgebildet sind (Blei et al. (2003), S. 994). Durch die linearität des Modelles kommt es zu overfitting, zusätzlich gibt es keine Möglichkeit nicht im Training enthaltenen Dokumenten eine Wahrscheinlichkeit zuzuweisen (Blei et al. (2003), S. 994).
Um Repräsentationen von Dokumenten und Wörtern austauschbar abzubilden, ist es notwendig mixture-models zu verwenden und somit den angeführten Limitationen von LSI und verwandten Ansätzen zu entgehen (Blei et al. (2003), S. 995).

Außerdem grenzt sich LDA von Hierarchical Latent Tree Analysis (HLTA) wie folgt ab, HLTA stellt Koinzidenz Muster explizit in Modellstrukturen dar (Chen at al. (2017), S. 1). HLTA ist somit eine Methode um Themen zu erkennen, indem ein Thema anhand von Wörtern identifiziert wird, die häufig in einem Thema und selten in einem anderen Thema auftreten (Liu et al. (2014), S. 1).



# Improved LDA


# Benchmarking & Visualisierung
Im Rahmen dieser Untersuchung werden die Ergebnisse der beiden LDA-Versionen (einfach und verbessert) hinsichtlich folgender Parameter analysiert:
* Vergleich der PyLDAvis-Visualisierung
* Vergleich der Kategorien, die übereinstimmen
* Vergleich der Übereinstimmungsrate
* Vergleich der Laufzeit
* Vergleich der Kohärenz pro Modell

Im Folgenden werden die Projekte in den jeweiligen Kapiteln zusammengefasst.
* Vergleichende Gegenüberstellung beider Methoden
* Herausforderungen und Optimierungen
* Schlussfolgerung

## Vergleich der Kategorie-Ermittlung
Der Vergleich erfolgt lediglich zwischen dem vorhergesagten Kategoriewert und dem bereits definierten Wert aus dem ursprünglichen Dataset, wobei die *short_description* und die *headline* berücksichtigt werden.

Eine höheren Anzahl an vorhergesagten Kategorien und eine höhere Kompetabilität führen zu einem besseren Ergebnis.

### PyLDAVis
Die resultierenden LDA-Parameter werden zunächst mithilfe der interaktiven Pyldavis-Bibliothek visualisiert, die sich in besonderem Maße für die LDA-Themenmodellierung eignet.

Die als Kreise dargestellten Themen $T$ werden hinsichtlich ihrer relativen Häufigkeit in dem Dokumenten-Corpus $D$ abgebildet. Die Größe der Kreise spiegelt somit die jeweilige Relevanz wider. Die Position der Kreise im Diagramm gibt Aufschluss über die Ähnlichkeit oder Differenzierung der einzelnen Themen zueinander. Durch Selektion eines Kreises werden die für das jeweilige Thema relevanten Hauptwörter angezeigt, was zu einer vertieften Einsicht in die Relevanz der Wörter in Bezug auf das Thema führt. 

Zudem ist eine Einschätzung derjenigen Wörter möglich, die für das Modell von Bedeutung sind, sowie eine Einschätzung der Wichtigkeit und Einflussnahme einzelner Wörter auf die Themen.

### Übereinstimmungs Vergleich
Im Folgenden wird die vorliegende Kategorie mit der prognostizierten Kategorie jedes LDA verglichen, wobei zusätzlich die jeweilige Vorhersage in Prozent angegeben wird. Dies dient der besseren Verständlichkeit.

Für eine weiterführende Analyse werden zudem die nicht zugeordneten Kategorien aufgeführt.

## Laufzeit Vergleich
Des Weiteren ist eine Gegenüberstellung des Laufs mit seiner Laufzeit sowie des erzielten Kohärenzscores beider Versionen von entscheidender Bedeutung. Dabei gilt, dass ein höherer Kohärenzscore eine höhere Qualität der berechneten Themen widerspiegelt und somit als besser zu bewerten ist.


## Allgemeiner Vergleich
Im Folgenden werden die beiden implementierten Methoden hinsichtlich der bislang noch nicht evaluierten Parameter miteinander verglichen.

Der zugrunde liegende Prozess jeder Methode war im Wesentlichen ähnlich:
1. Reinigen der Daten und vorbereitung für den Trainingsschritt
2. Trainieren des LDA
3. Evaluieren des Ergebnisses und Kontextverständnis jenes


### Simple LDA
**Pros:**
* Einfach zu implementieren
* Schnelles trainieren einfacher Datasets für "gute" Ergebnisse

**Cons:**
* Fehleranfällig
* Underfitting ist ein Problem
<br>

### Improved LDA
**Pros:**
* LDA Methode in verbesserter Variante
* Aufschlussreiche Informationen im Vergleich mit *simple_lda*
* Mehr Möglichkeiten den Prozess zu verbessern


**Cons:**
* Komplex und kompliziert die Methode des Papers zu reproduzieren
* Langsame Laufzeit auf einer lokalen Maschine

Daher erübrigt sich die implementierte Lösung für "improved_lda" bei der Ausbildung eines LDA, zumindest bei korrekter Aufbereitung der Daten.




# Citations

