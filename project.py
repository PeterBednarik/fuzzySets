import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def drop_everything_but_specified_coll(dataset:pd.DataFrame, columns:list) -> pd.DataFrame:
    for col in dataset.columns:
        if col not in columns:
            dataset.drop([col],axis=1,inplace=True)
    return dataset


def calculate_overall_satisfaction(data_frame:pd.DataFrame, list_of_cols:list) -> pd.DataFrame:
    # Check if the specified columns are present in the DataFrame
    missing_cols = set(list_of_cols) - set(data_frame.columns)
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} not found in the DataFrame.")

    # Create a new column "satisfaction" by calculating the average across specified columns
    data_frame['satisfaction'] = data_frame[list_of_cols].mean(axis=1)

    return data_frame


def take_user_input_two(interval_min:int, interval_max:int, prompt_message:str) -> tuple:
    limit = 3
    count = 0
    while limit >= count:
        count += 1
        user_input = input(f"{prompt_message}: ")

        # Split the input into two parts using comma as the delimiter
        input_values = user_input.split(',')

        # Check if there are exactly two values after splitting
        if len(input_values) == 2:
            try:
                # Convert the input values to integers
                num1, num2 = sorted(map(int, input_values))

                # Check if the values are within the specified interval
                if interval_min <= num1 <= interval_max and interval_min <= num2 <= interval_max:
                    return (num1, num2)
                else:
                    print(f"Values must be in the interval [{interval_min}, {interval_max}]")
            except ValueError:
                print("Invalid input. Please enter two integers separated by a comma.")
        else:
            print("Invalid input. Please enter two numbers separated by a comma.")


def take_user_input_four(interval_min:int, interval_max:int, prompt_message:str) -> tuple:
    limit = 3
    count = 0
    while limit >= count:
        count += 1
        user_input = input(f"{prompt_message}: ")

        # Split the input into four parts using commas as the delimiters
        input_values = user_input.split(',')

        # Check if there are exactly four values after splitting
        if len(input_values) == 4:
            try:
                # Convert the input values to integers
                num1, num2, num3, num4 = sorted(map(int, input_values))

                # Check if the values are within the specified interval
                if (
                    interval_min <= num1 <= interval_max and
                    interval_min <= num2 <= interval_max and
                    interval_min <= num3 <= interval_max and
                    interval_min <= num4 <= interval_max
                ):
                    return (num1, num2, num3, num4)
                else:
                    print(f"Values must be in the interval [{interval_min}, {interval_max}]")
            except ValueError:
                print("Invalid input. Please enter four integers separated by commas.")
        else:
            print("Invalid input. Please enter four numbers separated by commas.")


# Define y-values based on the fuzzy set type
def get_y_values(form):
    if form == 'L':
        return [1, 1, 0, 0]
    elif form == 'R':
        return [0, 0, 1, 1]
    elif form == 'T':
        return [0, 0, 1, 1, 0, 0]
    else:
        raise ValueError(f"Invalid type: {form}")


def get_single_plot(form:str, x:list) -> None:
    if form == 'L':
        color = 'red'
    elif form == 'R':
        color = 'blue'
    elif form == 'T':
        color = 'green'
    y = get_y_values(form)
    
    plt.plot(x, y, color=color)
    plt.fill_between(x, 0, y, color=color, alpha=0.2, label='Filled Area')
    plt.title('Fuzzy Set', fontsize=14)
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    # Customize grid lines to be present only in the area of x-values
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # Customize x-axis ticks to include specific values from your data
    plt.xticks(x, rotation=45, ha="right")
    plt.show()


def get_multiple_plot2(type_list, *args, labels=None):
    """
    Create a plot with multiple lines, each corresponding to a list argument.

    Parameters:
    type_list: List of types corresponding to each argument.
    *args: Variable number of lists to be plotted.
    labels: List of labels for each line.

    Returns:
    None
    """
    # Generate a color map for different line colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(args)))


    # Plot each line with a different color and corresponding y-values
    for i, (arg, form) in enumerate(zip(args, type_list)):
        color = colors[i]
        label = labels[i] if labels else f'Line {i+1}'
        y_values = get_y_values(form)
        plt.plot(arg, y_values, color=color, label=label)
        plt.fill_between(arg, 0, y_values, color=color, alpha=0.2, label=f'Filled Area {i+1}')

    # Set labels and title
    plt.title('Multiple Lines Plot', fontsize=14)
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)

    # Add grid
    plt.grid(True)

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()


def create_affiliation_column(type:str, breakpoints:list, col_name:str, data:pd.DataFrame) -> pd.DataFrame:
    new_column = f"affiliation_{col_name}"

    if type not in ['L', 'R', 'T']:
        raise ValueError("Invalid type. Type must be 'L' or 'R'.")

    if new_column in data.columns:
        raise ValueError(f"Column {new_column} already exists in the dataframe.")

    data[new_column] = 0.0  # Initialize the new column with zeros

    for row in range(len(data)):
        if type == 'L':
            if data[col_name][row] < breakpoints[0]:
                data.at[row, new_column] = 1
            elif data[col_name][row] > breakpoints[1]:
                data.at[row, new_column] = 0
            else:
                data.at[row, new_column] = (breakpoints[1] - data[col_name][row]) / (breakpoints[1] - breakpoints[0])   # L
        elif type == 'R':
            if data[col_name][row] < breakpoints[0]:
                data.at[row, new_column] = 0
            elif data[col_name][row] > breakpoints[1]:
                data.at[row, new_column] = 1
            else:
                data.at[row, new_column] = (data[col_name][row] - breakpoints[0]) / (breakpoints[1] - breakpoints[0])   # R
        elif type == 'T':
            # R = a and b, L = c and d
            if (data[col_name][row] < breakpoints[0]) or (data[col_name][row] > breakpoints[3]):
                data.at[row, new_column] = 0
            elif (data[col_name][row] > breakpoints[1]) and (data[col_name][row] < breakpoints[2]):
                data.at[row, new_column] = 1
            elif (data[col_name][row] > breakpoints[0]) and (data[col_name][row] < breakpoints[1]):
                data.at[row, new_column] = (data[col_name][row] - breakpoints[0]) / (breakpoints[1] - breakpoints[0])   # R
            else:
                data.at[row, new_column] = (breakpoints[3] - data[col_name][row]) / (breakpoints[3] - breakpoints[2])   # L
            

    return data


def calculate_and_assign_min(df:pd.DataFrame, col1:str, col2:str) -> pd.DataFrame:
    df['MIN'] = df.apply(lambda row: min(row[col1], row[col2]), axis=1)
    return df


if __name__ == '__main__':
    fuzzy_dataset = pd.read_csv('./test.csv')   # Load data
    fuzzy_dataset = fuzzy_dataset.iloc[:,1:]
    print(f'INITIAL dataset:\n{fuzzy_dataset.head()}')

    do_not_drop = ['Flight Distance', 'Ease of Online booking', 'Inflight wifi service', 'On-board service', 'Baggage handling', 'Leg room service', 'Food and drink']
    fuzzy_dataset = drop_everything_but_specified_coll(fuzzy_dataset, do_not_drop)
    print(f'Dataset with only specified col:\n{fuzzy_dataset.head()}')

    satisfaction_columns = ['Ease of Online booking', 'Inflight wifi service', 'On-board service', 'Baggage handling', 'Leg room service', 'Food and drink']
    fuzzy_dataset = calculate_overall_satisfaction(fuzzy_dataset, satisfaction_columns)
    print(f'Dataset with satisfaction col:\n{fuzzy_dataset.head()}')
    # Assuming your DataFrame is named df
    print(fuzzy_dataset['satisfaction'].describe())

    # # priklad z hodiny
    # fuzzy_dataset = pd.DataFrame({
    # 'Flight Distance': [39, 43, 28, 50, 21, 37.5, 10, 49, 36],
    # 'satisfaction': [100, 220, 130, 210, 500, 225, 190, 210, 190]
    # })
    max_value_distance = fuzzy_dataset['Flight Distance'].max()
    min_value_distance = fuzzy_dataset['Flight Distance'].min()
    max_value_satisfaction = fuzzy_dataset['satisfaction'].max()
    # mim_value_satisfaction = fuzzy_dataset['satisfaction'].min()    # priklad z hodiny
    mim_value_satisfaction = 0
    print(f'maximal flight distance is: {max_value_distance}, Min value is: {min_value_distance}')
    print(f'maximal satisfaction is: {max_value_satisfaction}, Min value is: {mim_value_satisfaction}\n')

    # clear the dataset to have only wanted columns
    fuzzy_dataset = drop_everything_but_specified_coll(fuzzy_dataset, ["Flight Distance", "satisfaction"])

    # -----------------------------------------------------

    # we can have 3 fuzzy sets for flight distance
    # and 3 fuzzy sets for satisfaction
    # 1. ask user which set he wants for 'flight distance' column {"small_distance"="L", "medium_distance"="T", "long_distance"="R"}
    user_input = input('Which flight distance set do you choose ("L", "R", "T"): ')
    user_input = user_input.strip().upper()
    if user_input not in ["L", "T", "R"]:
        print('\t\tINVALID INPUT -- EXIT')
        raise SystemExit(0)
    
    # 1.1 take a breakpoint values for coresponding set
    if user_input == "T":
         flight_distance_intervals = take_user_input_four(min_value_distance, max_value_distance, f'Give me 4 numbers for Flight Distance {user_input} fuzzy set: ')
    else:
        flight_distance_intervals = take_user_input_two(min_value_distance, max_value_distance, f'Give me 2 numbers for Flight Distance {user_input} fuzzy set: ')

    # 2. create an affiliation column for 'flight distance'
    fuzzy_dataset = create_affiliation_column(user_input, list(flight_distance_intervals), "Flight Distance", fuzzy_dataset)
    print(f'Affiliation to R fuzzy added to flight distance:\n{fuzzy_dataset.head()}')
    # 2.1 ask user if he wants to see how does FUZZY SET looks like / fuzzy setS?
    show_fuzzy = input('Do you want to show me the dataset? (Y/N): ')
    show_fuzzy = show_fuzzy.strip().upper()
    if show_fuzzy == "Y":
        if user_input == "T":
            my_x = [min_value_distance, flight_distance_intervals[0], flight_distance_intervals[1], flight_distance_intervals[2], flight_distance_intervals[3], max_value_distance]
        else:
            my_x = [min_value_distance, flight_distance_intervals[0], flight_distance_intervals[1], max_value_distance]
        get_single_plot(user_input, my_x)
    elif show_fuzzy != "N":
        print('\t\tINVALID INPUT -- EXIT?')
        raise SystemExit(0)
    print("\n-- Part for Flight Distance done --\n")    # info


    # 3. ask user which set he wants for 'satisfaction' column {"low_satisfaction"="L", "medium_satisfaction"="T", "big_satisfaction"="R"}
    user_input = input('Which satisfaction set do you choose ("L", "R", "T"): ')
    user_input = user_input.strip().upper()
    if user_input not in ["L", "T", "R"]:
        print('\t\tINVALID INPUT -- EXIT?')
        raise SystemExit(0)
    # 3.1 take a breakpoint values for coresponding set
    print('\n>>WARNING:: satisfaction have values between 0-5 so you shoul use this values')
    print('>>L-fuzzy Low_satisfaction: 0,1,2\n>>T-fuzzy medium_satisfaction: 1,2,3,4\n>>R-fuzzy big_satisfaction: 3,4,5\n')
    if user_input == "T":
         satisfaction_intervals = take_user_input_four(mim_value_satisfaction, max_value_satisfaction, f'Give me 4 numbers for Flight Distance {user_input} fuzzy set: ')
    else:
        satisfaction_intervals = take_user_input_two(mim_value_satisfaction, max_value_satisfaction, f'Give me 2 numbers for Flight Distance {user_input} fuzzy set: ')

    # 4. create an affiliation column for 'satisfaction'
    fuzzy_dataset = create_affiliation_column(user_input, list(satisfaction_intervals), "satisfaction", fuzzy_dataset)
    print(f'Affiliation to R fuzzy added to flight distance:\n{fuzzy_dataset.head()}')
    # 4.1 ask user if he wants to see how does FUZZY SET looks like / fuzzy setS?
    show_fuzzy = input('Do you want to show me the fuzzy set? (Y/N): ')
    show_fuzzy = show_fuzzy.strip().upper()
    if show_fuzzy == "Y":
        if user_input == "T":
            my_x = [mim_value_satisfaction, satisfaction_intervals[0], satisfaction_intervals[1], satisfaction_intervals[2], satisfaction_intervals[3], max_value_satisfaction]
        else:
            my_x = [mim_value_satisfaction, satisfaction_intervals[0], satisfaction_intervals[1], max_value_satisfaction]
        get_single_plot(user_input, my_x)
    elif show_fuzzy != "N":
        print('\t\tINVALID INPUT -- EXIT?')
        raise SystemExit(0)
    print("\n-- Part for satisfaction done --\n")    # info

    # now we have the final dataset with both affiliation columns
    # 5. create new column named 'MIN' = take the min from (affiliation_1, affiliation_2) columns
    affiliation_Flight_Distance_sum = fuzzy_dataset['affiliation_Flight Distance'].sum()
    affiliation_satisfaction_sum = fuzzy_dataset['affiliation_satisfaction'].sum()
    fuzzy_dataset = calculate_and_assign_min(fuzzy_dataset, 'affiliation_Flight Distance', 'affiliation_satisfaction')
    min_sum = fuzzy_dataset['MIN'].sum()
    print(f'Sum of the min column: {min_sum}, affiliation_Flight Distance sum: {affiliation_Flight_Distance_sum}')

    # 6. our_num = sum(df[min]):sum(df[affiliate_1])
    our_num = min_sum / affiliation_Flight_Distance_sum
    print(f'\n>>Value which we are finding the affiliation: {our_num}')

    # 7. create a final R fuzzy set, breakpoints=[0.5, 0.85] as final_R_set
    final_R_fuzzy_set = [0.5, 0.85]

    # 8. evaluate the probability of our_number in final_R_set
    if our_num < final_R_fuzzy_set[0]:
        final_probability = 0
    elif our_num > final_R_fuzzy_set[1]:
        final_probability = 1
    else:
        final_probability = (our_num-final_R_fuzzy_set[0])/(final_R_fuzzy_set[1]-final_R_fuzzy_set[0])
    print(f'\n>>Final probability is: {final_probability}')


    # # take user input for the specified fuzzy set
    # intervals_long_distance = take_user_input_two(min_value_distance, max_value_distance, 'Give me 2 numbers for long_distance')
    # # create affiliation column in dataset
    # fuzzy_dataset = create_affiliation_column("R", list(intervals_long_distance), "Flight Distance", fuzzy_dataset)
    # print(f'Affiliation to R fuzzy added to flight distance:\n{fuzzy_dataset.head()}')
    # fuzzy_dataset = create_affiliation_column("R", list(intervals_long_distance), "satisfaction", fuzzy_dataset)
    # print(f'Affiliation to R fuzzy added to satisfaction:\n{fuzzy_dataset.head()}')
    # --- FUZZY REPRESENTATION ---
    # intervals_small_distance = take_user_input_two(min_value_distance, max_value_distance, 'Give me 2 numbers for small_distance')
    # my_x_small = [min_value_distance, intervals_small_distance[0], intervals_small_distance[1], max_value_distance]
    # # get_single_plot('L', my_x_small)

    # intervals_medium_distance = take_user_input_four(min_value_distance, max_value_distance, 'Give me 4 numbers for medium_distance')
    # my_x_medium = [intervals_medium_distance[0], intervals_medium_distance[1], intervals_medium_distance[2], intervals_medium_distance[3]]
    # # get_single_plot('T', my_x_medium)

    # intervals_long_distance = take_user_input_two(min_value_distance, max_value_distance, 'Give me 2 numbers for long_distance')
    # my_x_long = [min_value_distance, intervals_long_distance[0], intervals_long_distance[1], max_value_distance]
    # # get_single_plot('R', my_x_long)
    
    # type_list = ['L', 'T', 'R']
    # labels = ["L-Form", "Triangle" ,"R-Form"]
    # get_multiple_plot2(type_list,my_x_small, my_x_medium, my_x_long, labels=labels)