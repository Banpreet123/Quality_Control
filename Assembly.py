import argparse
import ast
import re
from MySQL import MySQL
from collections import defaultdict
from TaggerClassifier import TaggerClassifier
from TagClassifier import TagClassifier
from PatternDetection_refactored import PatternDetection
import numpy as np
import pandas as pd
import os


class Application:
    """
    Master class of the Application
    """
    def __init__(self) -> None:
        self._connector = MySQL()                       # MySQL connector to call methods of MySQL class
        self.assignment_to_users = defaultdict(dict)    # dictionary to store the result of interval logs query
        self.tagger_classifier = TaggerClassifier()     # object of TaggerClassifier class
        self.tag_classifier = TagClassifier()           # object of TagClassifier class
        self.interval_logs_result = defaultdict(dict)   # result of interval logs
        self.krippendorff_result = defaultdict(dict)    # result of krippendorff alpha for a user
        self.agree_disagree_tags = defaultdict(dict)    # result of agreement/disagreement for each tag
        self.pattern_detection_result = defaultdict(dict) # result of interval logs
        self.assignment_to_user = defaultdict(dict)   
        self.user_history_dict = defaultdict(dict)
        self.pattern_detection = PatternDetection()
        self.assignment_to_teams = {}                  # dictionary that stores the result of getUserTeams function
        
    def __getIntervalLogs(self, tags, log_time=None) -> None:
        """
        Calculates Interval logs

        Args:
            tags (list): List of tags
        """
        # Populating the assignment_to_users hashmap based on the assignment_id and user_id
        for tag in tags:
            if tag.assignment_id in self.assignment_to_users:
                if tag.user_id in self.assignment_to_users[tag.assignment_id]:
                    self.assignment_to_users[tag.assignment_id][tag.user_id].append(tag)
                else:
                    self.assignment_to_users[tag.assignment_id][tag.user_id] = [tag]
            else:
                self.assignment_to_users[tag.assignment_id] = {tag.user_id: [tag]}
                
        # Calculating interval logs per assignment per user
        for assignment_id, users in self.assignment_to_users.items():
            for user,tags in users.items():
                self.interval_logs_result[assignment_id][user] = self.tagger_classifier.buildIntervalLogs(self.assignment_to_users[assignment_id][user])
                
        with open("data/Interval_logs.csv", "w") as f:
            f.write("Assignment_id,User_id,IL_result,Time,Number_of_Tags\n")
            for assignment_id, users in self.interval_logs_result.items():
                for user_id, results in users.items():
                    log_time_value = results[0]
                    number_of_tags = results[1]
                    if log_time is None or log_time_value >= log_time:
                        # Format IL_result and Time to 3 decimal places
                        il_result_formatted = "{:.3f}".format(log_time_value)
                        time_formatted = "{:.3f}".format(pow(2, log_time_value))
                        f.write(f"{assignment_id},{user_id},{il_result_formatted},{time_formatted},{number_of_tags}\n")
        print("Interval logs written to data/Interval_logs.csv")
        f.close()
    
    def __getUserHistory(self, user_history) -> None:
        """
        Calculates Interval logs

        Args:
            tags (list): List of tags
        """         
        for tag in user_history:
            if tag.assignment_id in self.user_history_dict:
                if tag.user_id in self.user_history_dict[tag.assignment_id]:
                    self.user_history_dict[tag.assignment_id][tag.user_id].append(tag)
                else:
                    self.user_history_dict[tag.assignment_id][tag.user_id] = [tag]
            else:
                self.user_history_dict[tag.assignment_id] = {tag.user_id: [tag]}

        # Calculating interval logs per assignment per user
        f = open("data/user_data.csv","w")
        f.write("User_id,Assignment_id,Question,Score,Review_Comment,Tag_Prompt,Tag_Value\n")
        for assignment_id, users in self.user_history_dict.items():
            for user,tags in users.items():                    
                for tag in tags:
                    # Clean 'question' by removing HTML tags and commas
                    cleaned_question = re.sub('<.*?>', '', tag.question).replace(',', '').replace('\n', '').replace('\r', '')
                    # Clean 'comments' by removing HTML tags and commas
                    cleaned_comments = re.sub('<.*?>', '', tag.comments).replace(',', '').replace('\n', '').replace('\r', '')
                    
                    # Concatenate the cleaned strings with other information and add it to the output string
                    output_string = f"{str(user)},{str(assignment_id)},{cleaned_question},{tag.answer_score},{cleaned_comments},{tag.prompt},{tag.value}\n"
                    f.write(output_string)
        print("User data written to data/user_data.csv")
        f.close()
  
    def __getKrippendorffAlpha(self, alpha=None):
        """
        Calculates krippendorf alpha value for each user
        """
        
        for assignment in self.assignment_to_teams:
            for team in self.assignment_to_teams[assignment].teams:
                data = []
                answers = defaultdict(set)
                users = []
                
                #collecting all the tag_prompt_ids for al answers of a team
                for user in self.assignment_to_teams[assignment].teams[team].users:
                    users.append(user)
                    for answer in self.assignment_to_teams[assignment].teams[team].users[user].answers:
                        for tag in self.assignment_to_teams[assignment].teams[team].users[user].answers[answer].tags:
                            answers[answer].add(tag)
                
                # collecting all the raters data for a single tag_prompt_id
                for answer in answers:
                    for tag in answers[answer]:
                        row = []
                        for user in self.assignment_to_teams[assignment].teams[team].users:
                            if answer not in self.assignment_to_teams[assignment].teams[team].users[user].answers:
                                row.append(np.nan)
                            elif tag not in self.assignment_to_teams[assignment].teams[team].users[user].answers[answer].tags:
                                row.append(np.nan)
                            else:
                                row.append(self.assignment_to_teams[assignment].teams[team].users[user].answers[answer].tags[tag].value)
                
                        data.append(row)
                if len(data[0])==1:
                    self.krippendorff_result[assignment][team] = {users[0]: np.nan}
                    continue
                data = np.array(data)
                
                #calculating krippendorff's alpha for all users in a team for an assignment
                self.krippendorff_result[assignment][team] = self.tagger_classifier.computeKrippendorffAlpha(data, users)
        
        #writing the krippendorff's alpha to a csv file if the alpha value is greater than the given alpha value
        f = open("data/krippendorff.csv", "w")
        f.write("Assignment_id,Team_id,User_id,Alphas\n")
        for assignment_id, teams in self.krippendorff_result.items():
            for team_id, users_alphas in teams.items():
                for user_id, alpha_value in users_alphas.items():
                    # Check if alpha is None or the alpha value is greater or equal than the given alpha threshold
                    if alpha is None or alpha_value >= alpha:
                        # Format Alphas to 3 decimal places if it's a float, otherwise write 'nan'
                        alpha_formatted = "{:.3f}".format(alpha_value) if isinstance(alpha_value, float) else "nan"
                        f.write(f"{assignment_id},{team_id},{user_id},{alpha_formatted}\n")
        print("Krippendorff's alpha written to data/krippendorff.csv")
        f.close()
     
    def __calculateAgreementDisagreement(self):
        for assignment in self.assignment_to_teams:
            for team in self.assignment_to_teams[assignment].teams:
                data = []
                answers = defaultdict(set)
                
                #collecting all the tag_prompt_ids for al answers of a team
                for user in self.assignment_to_teams[assignment].teams[team].users:
                    for answer in self.assignment_to_teams[assignment].teams[team].users[user].answers:
                        for tag in self.assignment_to_teams[assignment].teams[team].users[user].answers[answer].tags:
                            answers[answer].add(tag)
                
                # collecting all the raters data for a single tag_prompt_id
                for answer in answers:
                    for tag in answers[answer]:
                        row = []
                        for user in self.assignment_to_teams[assignment].teams[team].users:
                            if answer not in self.assignment_to_teams[assignment].teams[team].users[user].answers:
                                row.append(None)
                            elif tag not in self.assignment_to_teams[assignment].teams[team].users[user].answers[answer].tags:
                                row.append(None)
                            else:
                                row.append(self.assignment_to_teams[assignment].teams[team].users[user].answers[answer].tags[tag])
                
                        data.append(row)
                if len(data[0])==0:
                    self.agree_disagree_tags[assignment][team] = np.nan
                    continue
                data = np.array(data)
                
                #calculating agreement/disagreement of all tags
                self.agree_disagree_tags[assignment][team] = self.tag_classifier.calculateAgreementDisagreement(data)
        
        f = open("data/tags.csv","w")
        f.write("Assignment_id,team_id,answer_id,tag_prompt_id,value,fraction\n")
        for i in self.agree_disagree_tags:
            for j in self.agree_disagree_tags[i]:
                for k in self.agree_disagree_tags[i][j]:
                    for l in self.agree_disagree_tags[i][j][k]:
                        f.write(str(i)+","+str(j)+","+str(k)+","+str(l)+","+str(self.agree_disagree_tags[i][j][k][l][0])+","+str(self.agree_disagree_tags[i][j][k][l][1])+"\n")
        f.close()        
        
    def assignTaggerReliability(self, log_time=None, alpha=None, lmin=5, lmax=30, minrep=15):
        """
        Function used to compute Interval Logs, Krippendorff Alpha and Pattern detection
        """
        # Interval logs
        self.tags = self._connector.getAnswerTags()
        self.__getIntervalLogs(self.tags, log_time)
    
        # Krippendorff alpha
        self.assignment_to_teams = self._connector.getUserTeams()
        self.__getKrippendorffAlpha(alpha)

        # # Pattern Detection
        self.__getPatternResults(self.tags, lmin, lmax, minrep)

        # User data
        self.user_history = self._connector.getUserHistory()
        self.__getUserHistory(self.user_history)
        
    def assignTagReliability(self):
        """
        Function used to compute Agreement/Disagreement of tags
        """
        self.__calculateAgreementDisagreement()

    def __getPatternResults(self, tags, lmin=5, lmax=30, minrep=15) -> None:
        """
        Calculates pattern detection results

        Args:
            tags (list): List of tags
        """
        # Populating the assignment_to_users hashmap
        for tag in tags:
            if tag.assignment_id in self.assignment_to_user:
                if tag.user_id in self.assignment_to_user[tag.assignment_id]:
                    self.assignment_to_user[tag.assignment_id][tag.user_id].append(tag)
                else:
                    self.assignment_to_user[tag.assignment_id][tag.user_id] = [tag]
            else:
                self.assignment_to_user[tag.assignment_id] = {tag.user_id: [tag]}
        
        # Calculating pattern detection results for each assignment and user
        for assignment_id, users in self.assignment_to_user.items():
            for user, tags in users.items():
                pattern_results = self.pattern_detection.PTV(tags, lmin, lmax, minrep)
                self.pattern_detection_result[assignment_id][user] = pattern_results
        
        # Writing the pattern detection results to a file
        with open("data/Pattern_recognition.txt", "w") as f:
            f.write("Assignment_id/User_id/PD_result/Pattern/Repetition\n")
            for assignment_id, users in self.pattern_detection_result.items():
                for user, patterns in users.items():
                    for pattern, count in patterns:
                        if pattern != "Not_found":
                            f.write(f"{assignment_id}/{user}/Found/{pattern}/{count}\n")
                        else:
                            f.write(f"{assignment_id}/{user}/Not_found\n")
        print("Pattern recognition results written to data/Pattern_recognition.txt")

    
    def combine_csv_results(self, output_file):
        # Read the CSV files into DataFrames
        interval_logs_df = pd.read_csv("data/Interval_logs.csv")
        krippendorff_df = pd.read_csv("data/krippendorff.csv")
        pattern_results_df = pd.read_csv("data/Pattern_recognition.txt", sep="/", header=0, names=["Assignment_id", "User_id", "PD_result", "Pattern", "Repetition"])

        # Merge the DataFrames on 'Assignment_id' and 'User_id'
        merged_df = interval_logs_df.merge(krippendorff_df, on=['Assignment_id', 'User_id'])
        merged_df = merged_df.merge(pattern_results_df, on=['Assignment_id', 'User_id'])

        # Ensure the 'Time' column is present after the merge
        print(merged_df.columns)

        # Replace 'N/A' with a space in the entire DataFrame
        merged_df.replace('N/A', ' ', inplace=True)
        
        # Replace '-1' with a space in the 'FAST TAGGING LOG VALUES' column
        merged_df['IL_result'] = merged_df['IL_result'].apply(lambda x: ' ' if x == -1 else x)

        # Replace pattern strings like "('-1', '1', '1', '-1', '1', '1')" with "NYYNYY"
        merged_df['Pattern'] = merged_df['Pattern'].apply(
            lambda x: ''.join(['Y' if num == '1' else 'N' for num in ast.literal_eval(x)]) 
                    if isinstance(x, str) and x.startswith('(') else x
        )
            
        # Adding the 'Number of Tags Assigned' column using 'team_id'
        merged_df['Number_of_Tags_Assigned'] = merged_df['Team_id'].apply(lambda team_id: self._connector.getAnswerCountTimesThree(team_id))


        # Adjust the column order and rename as needed
        merged_df = merged_df[['User_id', 'Assignment_id', 'Team_id', 'IL_result', 'Time', 'Alphas', "Number_of_Tags", "Number_of_Tags_Assigned",  'PD_result', 'Pattern', 'Repetition']]
        merged_df.columns = ['USER ID', 'ASSIGNMENT ID', 'TEAM ID', 'FAST TAGGING LOG VALUES', 'FAST TAGGING SECONDS', 'ALPHA VALUES', 'NUMBER OF TAGS SET', 'NUMBER OF TAGS ASSIGNED', 'PATTERN FOUND OR NOT', 'PATTERN', 'PATTERN REPETITION']

        #Ensuring that patterns of a single user appear in the same line
        agg_funcs = {'ASSIGNMENT ID': 'min','TEAM ID':'min','FAST TAGGING LOG VALUES' :'min', 'FAST TAGGING SECONDS':'min','ALPHA VALUES':'min','NUMBER OF TAGS SET':'min',
             'NUMBER OF TAGS ASSIGNED':'min','PATTERN FOUND OR NOT':'first', 'PATTERN': lambda x: x.tolist(),'PATTERN REPETITION': lambda x: x.tolist()}
        # Group by 'id' and aggregate selected columns
        result_df = merged_df.groupby('USER ID', as_index=False).agg(agg_funcs)

        #removing 'nan' values for users where no pattern was found
        rows_not_found = result_df[result_df['PATTERN FOUND OR NOT'] == 'Not_found'].index
        columns_to_blank = ['PATTERN', 	'PATTERN REPETITION']  
        result_df.loc[rows_not_found, columns_to_blank] = ''

        #removing single quotes for each pattern found inside the array
        result_df['PATTERN'] = result_df['PATTERN'].replace({'["\']': ''}, regex=True) 

        #function that calculates the result of pattern length * pattern repetition
        def calculate_score(row):
            # Split the columns
            patterns = row['PATTERN']
            repetitions = row['PATTERN REPETITION']

            # Check if 'PATTERN' and 'PATTERN REPETITION' are not None and not empty lists
            if patterns is not None and repetitions is not None and patterns and repetitions:
                total_score = sum(len(pattern) * repetition for pattern, repetition in zip(patterns, repetitions))
                return total_score if total_score != 0 else None
            else:
                return None        

        # Apply the function 
        result_df['TOTAL COMMENTS TAGGED WITH PATTERNS'] = result_df.apply(calculate_score, axis=1)

        # Handling any Nan values
        rows = result_df[result_df['PATTERN FOUND OR NOT'] == 'Not_found'].index
        result_df.loc[rows, 'TOTAL COMMENTS TAGGED WITH PATTERNS'] = ''

        # Write the combined results to a new CSV file
        output_path = f"data/{output_file}"
        result_df.to_csv(output_path, index=False, na_rep=' ')

        print(f"Combined CSV created successfully as {output_path}")



if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    parser = argparse.ArgumentParser(description="Run the Application with specific parameters.")
    parser.add_argument('--log_time_min', type=float, default=None, help="Filtering value for log time.")
    parser.add_argument('--alpha_min', type=float, default=None, help="Filtering value for krippendorff alpha.")
    parser.add_argument('--min_pattern_len', type=int, default=5, help="Minimum value for pattern detection.")
    parser.add_argument('--max_pattern_len', type=int, default=30, help="Maximum value for pattern detection.")
    parser.add_argument('--min_pattern_rep', type=int, default=15, help="Minimum repetition value for pattern detection.")
    args = parser.parse_args()

    app = Application()
    app.assignTaggerReliability(args.log_time_min, args.alpha_min, args.min_pattern_len, args.max_pattern_len, args.min_pattern_rep)
    app.combine_csv_results('Combined_Results.csv')
