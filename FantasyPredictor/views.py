from django.shortcuts import render;
import pandas as pd
import numpy as np
import pulp
import json
import matplotlib.pyplot as plt
import os
from ast import literal_eval
from IPython.display import display
from django.shortcuts import HttpResponse

def home(request):
    return render(request, "home.html")

def predict(request):
    return render(request, "predict.html") 

def result(request):
    #Input
    team1 = str(request.GET['t1'])
    team2 = str(request.GET['t2'])

    result, team1_colname, team2_colname = calculate(request, team1, team2)
    selected_col = result.iloc[:,[0,1,2,3]]    

    # parsing the DataFrame in json format.
    json_records = selected_col.reset_index().to_json(orient ='records')
    data = []
    data = json.loads(json_records)
    context = {'d': data}
  
    bar_chart(result)
    pie_chart(result, team1_colname, team2_colname)
    return render(request, 'result.html', context)

def calculate(request, team1, team2):
    #Data path selection
    def getCsvPath(team1, team2):
        if team1 == team2: return 'Invalid Input'
        DATA_PATH = '/home/vishal/Documents/Fantasy-Prediction/data'
        team_mapping = {'CHENNAI SUPER KINGS': 'CSK', 
                        'DELHI CAPITALS': 'DC',
                        'KOLKATA KNIGHT RIDERS': 'KKR', 
                        'MUMBAI INDIANS': 'MI',  
                        'PUNJAB KINGS': 'KXIP', 
                        'RAJASTHAN ROYALS': 'RR', 
                        'ROYAL CHALLENGERS BANGLORE': 'RCB', 
                        'SUN RISERS HYDERABAD': 'SRH'
                        }
        csv_path = os.path.join(DATA_PATH, team_mapping[team1] + '_vs_' + team_mapping[team2] + '.csv')
        return csv_path


    #Loading csv file 
    data = pd.read_csv(getCsvPath(team1, team2), converters={"Match_points": literal_eval})


    #Exponentially weighted average computataion
    def compute_weighted_points(points_vector, alpha = 0.20):
        weights = np.exp(list(reversed(np.array(range(1, len(points_vector)+1))*alpha * -1)))
        exponential_weighted_average = np.average(np.array(points_vector), weights = weights)
        return exponential_weighted_average

    data['weighted_player_points'] = data['Match_points'].apply(compute_weighted_points)


    #Converts categorial data into indicator variable
    players_data = pd.get_dummies(data, columns=["Role", "Team"])


    # Initialize the optimization Problem 
    prob = pulp.LpProblem('Dreamteam', pulp.LpMaximize)


    selection_decision_variables = []


    for row in players_data.itertuples(index=True):
        variable_name = 'x_{}'.format(str(row.Index))
        variable = pulp.LpVariable(variable_name, lowBound = 0, upBound = 1, cat = 'Integer' ) 
        selection_decision_variables.append({"pulp_variable":variable, "Player": row.Player})
 
    selection_decision_variables_df = pd.DataFrame(selection_decision_variables)

    merged_players_data_df = pd.merge(players_data, selection_decision_variables_df, on = "Player")
    merged_players_data_df["pulp_variable_name"] = merged_players_data_df["pulp_variable"].apply(lambda x: x.name)
    


    total_points = pulp.lpSum(merged_players_data_df["weighted_player_points"] * selection_decision_variables_df["pulp_variable"])
    prob += total_points


    #Wicket keeper constraints Minimum 1 and Maximum 4
    total_keepers = pulp.lpSum(merged_players_data_df["Role_WK"] * selection_decision_variables_df["pulp_variable"])
    prob += (total_keepers >= 1)
    prob += (total_keepers <= 4)

    #Batsman constraints Minimum 3 and Maximum 6
    total_batsmen = pulp.lpSum(merged_players_data_df["Role_BAT"] * selection_decision_variables_df["pulp_variable"])
    prob += (total_batsmen >= 3)
    prob += (total_batsmen <= 6)

    #All Rounders constraints Minimum 1 and Maximum 4
    total_allrounders = pulp.lpSum(merged_players_data_df["Role_ALL"] * selection_decision_variables_df["pulp_variable"])
    prob += (total_allrounders >= 1)
    prob += (total_allrounders <= 4)

    #Bowlers constraints Minimum 3 and Maximum 6
    total_bowlers = pulp.lpSum(merged_players_data_df["Role_BALL"] * selection_decision_variables_df["pulp_variable"])
    prob += (total_bowlers >= 3)
    prob += (total_bowlers <= 6)

    #Maximum of 11 players selected from both the teams
    total_players = pulp.lpSum(selection_decision_variables_df["pulp_variable"])
    prob += (total_players == 11)

    #Maximum fantasy budget of 100
    total_cost = pulp.lpSum(merged_players_data_df["Cost"] * selection_decision_variables_df["pulp_variable"])
    prob += (total_cost <= 100)

    #Can't pick more than 7 players from the same team
    team1_colname = merged_players_data_df.columns[8]
    team2_colname = merged_players_data_df.columns[9]

    total_team1 = pulp.lpSum(merged_players_data_df[team1_colname] * selection_decision_variables_df["pulp_variable"])
    prob += (total_team1 <= 7)

    total_team2 = pulp.lpSum(merged_players_data_df[team2_colname] * selection_decision_variables_df["pulp_variable"])
    prob += (total_team2 <= 7)

    #assert len(pulp.listSolvers(onlyAvailable=True)) > 0, "solvers not installed correctly - check - https://www.coin-or.org/PuLP/main/installing_pulp_at_home.html"
    prob.solve()


    solutions_df = pd.DataFrame(
        [
            {
                'pulp_variable_name': v.name, 
                'value': v.varValue
            }
            for v in prob.variables()
        ]
    )


    result = pd.merge(merged_players_data_df, solutions_df, on = 'pulp_variable_name')
    result = result[result['value'] == 1].sort_values(by = 'weighted_player_points', ascending = False)
    
    return result, team1_colname, team2_colname

def bar_chart(result):

    data = {'Role_ALL': result[result['Role_ALL'] == 1]['Role_ALL'].count(),
            'Role_BALL': result[result['Role_BALL'] == 1]['Role_BALL'].count(),
            'Role_BAT': result[result['Role_BAT'] == 1]['Role_BAT'].count(),
            'Role_WK': result[result['Role_WK'] == 1]['Role_WK'].count()
            }

    roles = list(data.keys())
    values = list(data.values())
    
    # fig = plt.figure(figsize = (10, 10))
    
    # # creating the bar plot
    plt.bar(roles, values, color ='blue', width = 0.4)
    
    plt.xlabel("ROLE")
    plt.ylabel("NO. OF PLAYERS")
    plt.title("PLAYERS TYPE")
    #plt.show()

    plt.savefig('static/my_plot.png')
    #return render(request, "charts.html")
    plt.close()
    return

def pie_chart(result, team1_colname, team2_colname):
    data = {team1_colname: result[result[team1_colname] == 1][team1_colname].count(),
            team2_colname: result[result[team2_colname] == 1][team2_colname].count()}
    plt.pie(data.values(), labels=data.keys())
    plt.title("TEAM DISTRIBUTION")
    plt.savefig('static/my_pie.png')
    plt.close()
    return

def charts(request):
    return render(request, "charts.html")
