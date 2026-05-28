# 🏈 InteliNFL/InteliCFB

InteliNFL is a football analytics dashboard for NFL and college football decision intelligence. The app uses play-by-play data and interactive visualizations to help users explore games, quarterbacks, play calling, win probability, and team performance.

I built this project because football data is extremely detailed, but it can be hard to understand when it is only shown in raw tables. InteliNFL turns that data into a dashboard that is easier to filter, analyze, and use for football decision-making.

One of the most interesting parts of this project was supporting both NFL and college football data. Since those data ecosystems are different, I had to think carefully about how to normalize metrics and present the information clearly across both leagues.

## 📦 Technologies

- Python
- Streamlit
- pandas
- NumPy
- Plotly
- nfl-data-py
- CFBD API

## ✨ Features

- NFL and college football league toggle
- Game Analyzer
- QB Analyzer
- Play Calling analysis
- Win Probability views
- Team Profiles
- Downloadable filtered game data
- Interactive charts and dashboard views

## 🧠 The Process

I started by connecting the app to football data sources and figuring out what information would be most useful to analyze. Once the data was available, I worked on cleaning and organizing it so that both NFL and college football views could work inside the same dashboard.

After that, I built the app into different analysis tabs. Each tab focuses on a specific question, such as how a quarterback performed, how a game developed, or how a team approached play calling. I wanted the dashboard to feel focused instead of overwhelming, so I used filters, charts, and organized views to make the data easier to understand.

A major part of the process was thinking about how analytics should support decisions. The dashboard needed to do more than show numbers — it needed to help users quickly understand what the numbers mean.

## 📚 What I Learned

This project taught me how to work with sports data from multiple sources. NFL and college football datasets can have different structures and assumptions, so I learned how important normalization is when building analytics tools.

I also learned more about dashboard storytelling. A chart is only helpful if the user can understand what it shows, why it matters, and how it connects to the larger football context.

## 🔧 How It Can Be Improved

- Add stronger caching for faster performance
- Add more advanced filters
- Add model-based predictions
- Add user-saved dashboards
- Add deeper player comparison tools
- Improve mobile layout
- Add more historical trend views

## 🚀 Running the Project

```bash
git clone <repo-url>
cd "InteliNFL"

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# Optional: required for college football views
export CFBD_API_KEY=your_key

streamlit run app.py
```
