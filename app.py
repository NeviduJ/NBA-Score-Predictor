import gradio as gr
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from catboost import CatBoostRegressor
import tensorflow as tf

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

def predict(teamName, quarter, fieldGoalsMade, fieldGoalsAttempted, threePointersMade, threePointersAttempted, freeThrowsMade, freeThrowsAttempted, reboundsOffensive, reboundsDefensive, reboundsTotal, assists, steals, blocks, turnovers, foulsPersonal, points, plusMinusPoints):
    
    data = [teamName, quarter, fieldGoalsMade, fieldGoalsAttempted, threePointersMade, threePointersAttempted, freeThrowsMade, freeThrowsAttempted, reboundsOffensive, reboundsDefensive, reboundsTotal, assists, steals, blocks, turnovers, foulsPersonal, points, plusMinusPoints]
    column_names = ['teamName', 'quarter', 'fieldGoalsMade', 'fieldGoalsAttempted', 'threePointersMade', 'threePointersAttempted', 'freeThrowsMade', 'freeThrowsAttempted', 'reboundsOffensive', 'reboundsDefensive', 'reboundsTotal', 'assists', 'steals', 'blocks', 'turnovers', 'foulsPersonal', 'points', 'plusMinusPoints']
    # print(df)

    df = pd.DataFrame([data], columns= column_names)

    # print(df_home)

    df_main = pd.read_csv("2022_2023_NBA_Season_Quarterly_Data.csv")
    df_main = df_main[(df_main['quarter'] == quarter)]

    # df_main = df_main[df_main['teamName'].isin(team_names)]

    # df_main = df_main.dropna()
    # df_main = df_main.drop_duplicates()

    # df_main = df_main.pivot(index=['gameId', 'teamName', 'finalPoints'], columns='quarter', values=['fieldGoalsMade', 'fieldGoalsAttempted', 'threePointersMade', 'threePointersAttempted', 'freeThrowsMade', 'freeThrowsAttempted', 'reboundsOffensive', 'reboundsDefensive', 'reboundsTotal', 'assists', 'steals', 'blocks', 'turnovers', 'foulsPersonal', 'points', 'plusMinusPoints'])
    # df_main.columns = [f'{feature}_{inning}' for feature, inning in df_main.columns]
    # df_main = df_main.reset_index()
    df_main = df_main.drop(columns=['gameId', 'teamId', 'teamTricode', 'finalPoints'])
    df_main = pd.get_dummies(df_main, columns=['teamName'])

    # # df = pd.DataFrame([data], columns=["Team_Name", "Opposition_Team", "Inning", "Home/Away", "Hits", "Opp_Hits", "Errors", "Runs", "Opp_Runs", "LOB"])
    
    # pivoted_df = df.pivot(index=['teamName'], columns='quarter', values=['fieldGoalsMade', 'fieldGoalsAttempted', 'threePointersMade', 'threePointersAttempted', 'freeThrowsMade', 'freeThrowsAttempted', 'reboundsOffensive', 'reboundsDefensive', 'reboundsTotal', 'assists', 'steals', 'blocks', 'turnovers', 'foulsPersonal', 'points', 'plusMinusPoints'])
    # pivoted_df.columns = [f'{feature}_{inning}' for feature, inning in pivoted_df.columns]
    # # print(pivoted_df_home)
    # pivoted_df = pivoted_df.reset_index()

    df = pd.get_dummies(df, columns=['teamName'])
    df = df.reindex(columns=df_main.columns, fill_value=0)

    df = df.astype(int)
    print(df)

    # return

    # print(len(df.columns))
    if quarter == 1:
        model = tf.keras.models.load_model('ANNR_ts_q1_exp1_model.keras')
    elif quarter ==2:
        model = tf.keras.models.load_model('ANNR_ts_q2_exp1_model.keras')
    elif quarter ==3:
        model = tf.keras.models.load_model('ANNR_ts_q3_exp1_model.keras')
    


    # with open('pca_model4.pkl', 'rb') as f:
    #     pca = pickle.load(f)

    # with open('label_encoder_teams_xgbr1_exp3.pkl', 'rb') as f:
    #     label_encoder = pickle.load(f)
    
    # print(pivoted_df_home)

    # df = pca.transform(df)
    # return
    score= model.predict(df)
    print(score)
    score = [item for sublist in score for item in sublist]
    print(score)
    score = np.round(score[0],1)
    print(score)
    if score < 0:
        score = np.clip(score, a_min=0, a_max=None)
        # return score_1
    
    print(score)
    return score

team_names = ['Knicks',
 'Celtics',
 'Lakers',
 'Warriors',
 'Hornets',
 'Nets',
 'Bucks',
 'Nuggets',
 'Pacers',
 'Raptors',
 'Mavericks',
 'Heat',
 'Trail Blazers',
 '76ers',
 'Timberwolves',
 'Suns',
 'Hawks',
 'Rockets',
 'Clippers',
 'Bulls',
 'Pelicans',
 'Kings',
 'Wizards',
 'Jazz',
 'Magic',
 'Thunder',
 'Cavaliers',
 'Spurs',
 'Grizzlies',
 'Pistons',
 "Maccabi Ra'anana",
 'Flamengo',
 'Baloncesto']

#['fieldGoalsMade', 'fieldGoalsAttempted', 'threePointersMade', 'threePointersAttempted', 'freeThrowsMade', 'freeThrowsAttempted', 'reboundsOffensive', 'reboundsDefensive', 'reboundsTotal', 'assists', 'steals', 'blocks', 'turnovers', 'foulsPersonal', 'points', 'plusMinusPoints']

with gr.Blocks() as demo:
    # gr.Image("../Documentation/Context Diagram.png", scale=2)
    # gr(title="Your Interface Title")
    gr.Markdown("""
                <center> 
                <span style='font-size: 50px; font-weight: Bold; font-family: "Graduate", serif'>
                NBA Score Predictor 
                </span>
                </center>
                """)
    # gr.Markdown("""
    #             <center> 
    #             <span style='font-size: 30px; line-height: 0.1; font-weight: Bold; font-family: "Graduate", serif'>
    #             Admin Dashboard 
    #             </span>
    #             </center>
    #             """)
    with gr.Row():
        with gr.Column():
            teamName = gr.Dropdown(choices= team_names, max_choices= 1, label="Team Name", scale=1)

        with gr.Column():
            quarter = gr.Number(None, label="Quarter", maximum = 3, scale=1)
        
        with gr.Column():
            fieldGoalsMade = gr.Number(None, label="Field Goals Made (FGM)", scale=1)

    with gr.Row():
        with gr.Column():
            threePointersMade = gr.Number(None, label="3 Pointers Made (3PM)", scale=1)

        with gr.Column():
            fieldGoalsAttempted = gr.Number(None, label="Field Goals Attempted (FGA)", scale=1)

        with gr.Column():
            threePointersAttempted = gr.Number(None, label="3 Pointers Attempted (3PA)", scale=1)
    
    with gr.Row():
        with gr.Column():
            freeThrowsMade = gr.Number(None, label="Free Throws Made (FTM)", scale=1)
        
        with gr.Column():
            freeThrowsAttempted = gr.Number(None, label="Free Throws Attempted (FTA)", scale=1)

        with gr.Column():
            reboundsDefensive = gr.Number(None, label="Rebounds Defensive (DREB)", scale=1)

    with gr.Row():
        with gr.Column():
            reboundsOffensive = gr.Number(None, label="Rebounds Offensive (OREB)", scale=1)
    
        with gr.Column():
            reboundsTotal = gr.Number(None, label="Rebounds Total (REB)", scale=1)
        
        with gr.Column():
            assists = gr.Number(None, label="Assists (AST)", scale=1)

    with gr.Row():
        with gr.Column():
            steals = gr.Number(None, label="Steals (STL)", scale=1)
    
        with gr.Column():
            turnovers = gr.Number(None, label="Turnovers (TO)", scale=1)
        
        with gr.Column():
            foulsPersonal = gr.Number(None, label="Personal Fouls (PF)", scale=1)

    with gr.Row():
        with gr.Column():
            blocks = gr.Number(None, label="Blocks (BLK)", scale=1)

        with gr.Column():
            points = gr.Number(None, label="Points (PTS)", scale=1)

        with gr.Column():
            plusMinusPoints = gr.Number(None, label="+/- Points (+/-)", scale=1)

    with gr.Row():
        predict_btn = gr.Button(value="Predict Score", size = 'sm')

    with gr.Row():
        with gr.Column():
            final_score_away1 = gr.Textbox(label="Predicted Score", scale=1)

    predict_btn.click(predict, inputs=[teamName, quarter, fieldGoalsMade, fieldGoalsAttempted, threePointersMade, threePointersAttempted, freeThrowsMade, freeThrowsAttempted, reboundsOffensive, reboundsDefensive, reboundsTotal, assists, steals, blocks, turnovers, foulsPersonal, points, plusMinusPoints], outputs=final_score_away1)
    # predict_btn.click(predict, inputs=[opp_team, inning, opp_venue, opp_hits, opp_errors, opp_lob, opp_runs, team, runs, hits], outputs=final_score_home1)

    # predict_btn.click(predict_2, inputs=[team, inning, venue, hits, errors, lob, runs, opp_team, opp_runs, opp_hits], outputs=final_score_away2)
    # predict_btn.click(predict_2, inputs=[opp_team, inning, opp_venue, opp_hits, opp_errors, opp_lob, opp_runs, team, runs, hits], outputs=final_score_home2)

demo.launch(inbrowser=True)