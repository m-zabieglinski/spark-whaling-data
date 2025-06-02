# Spark Whaling Data
This repository contains a project concering exploration and trasnformation of American Offshore Whaling Logbook Data.
<br><br>
The data was obtained from https://whalinghistory.org/av/logs/aowl/ in October 2024
# Requirements
The project is split into 2 parts:
- Jupyter Notebooks with Almond:
    - Scala 2.11.x kernel
    - Spark 2.4.x
    - plotly-almond 0.8.5
- Main application code:
    - Scala 3.4.2
    - Spark 3.5.1
    - sbt
# Contents
The 2 project parts are interactable in following ways:
- Jupyter Notebooks:
    - exploration.ipynb
        - where the data was initially explored
        - generated the 2 plotly reports in html, browsable inside
    - species_predictor.ipynb
        - where the different machine learning models to label unidentified whales where initially tried out
        - random forest was chosen as the best suited for the task
    - Annual-whale-observations-per-species.html
        - interactive plotly report
    - Annual-whale-observations.html
        - interactive plotly report
- Main application code:
    - Main.scala
        - contains end-to-end random forest multi-model Spark pipeline for labeling unidentified whales
        - run with `sbt run` command from the repository
# Contact
https://m-zabieglinski.github.io/
<br>
mikolaj.zabieglinski [at] gmail.com