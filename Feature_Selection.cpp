#include <iostream>      // For standard input and output.
#include <fstream>       // For file handling.
#include <sstream>       // For string stream operations.
#include <vector>        // For using vectors.
#include <string>        // For string operations.
#include <cmath>         // For mathematical functions (e.g., sqrt, pow).
#include <limits>        // For handling infinite values.
#include <algorithm>     // For find function.

using namespace std;

int ALGORITHM_TYPE;     // Global variable to store the user's algorithm choice.

// Function declarations.
void input_data_from_file(vector<vector<double>>*, string);
void feature_search_forward(vector<vector<double>>);
void feature_search_backward(vector<vector<double>>);
double leave_one_out_cross_validation(vector<vector<double>>, vector<int>, int);

int main() {
    ios::sync_with_stdio(0);        // Fast input and output.

    vector<vector<double>> data;    // 2D vector to store dataset values.

    // Prompt user to enter the dataset filename.
    cout << "Welcome to Nelson Tran's Feature Selection Algorithm." << '\n'
         << "Type the name of the file to test: ";
    string file_name; getline(cin, file_name);
    cout << '\n';
    input_data_from_file(&data, file_name);

    // Prompt user to select the algorithm type.
    cout << "Type the number of the algorithm you want to run." << '\n'
         << "     1) Forward Selection" << '\n'
         << "     2) Backward Elimination" << '\n' << '\n';
    cin >> ALGORITHM_TYPE;
    cout << '\n';

    // Execute the selected algorithm.
    if (ALGORITHM_TYPE == 1) {
        feature_search_forward(data);
    }
    else if (ALGORITHM_TYPE == 2) {
        feature_search_backward(data);
    }

    return 0;
}

// Reads dataset from a file and stores it into a 2D vector.
void input_data_from_file(vector<vector<double>>* data, string file_name) {
    ifstream file(file_name);               // Open file.
    if (!file) {
        cerr << "Error: Could not open file!" << endl;
        return;
    }

    string line_in_file;
    while (getline(file, line_in_file)) {   // Reads file line by line.
        istringstream iss(line_in_file);    // Converts line into stream of inputs.
        vector<double> line;
        double num_from_file;

        while (iss >> num_from_file) {      // Extract numbers from the line.
            line.push_back(num_from_file);
        }

        if (!line.empty()) {
            data->push_back(line);          // Data gets updated with a line of numbers.
        }
    }

    file.close();
    return;
}

// Forward Selection Algorithm: Adds one feature at a time.
void feature_search_forward(vector<vector<double>> data) {
    vector<int> full_set_of_features;
    for (int i = 1; i < data[0].size(); i++) {  // Create a full set with all features.
        full_set_of_features.push_back(i);
    }

    // Tell user information about the data.
    cout << "This dataset has " << data[0].size() - 1 << " features (not including the class attribute), with " << data.size() << " instances." << '\n'
         << "Running nearest neighbor with all " << data[0].size() - 1 << " features, using \"leaving-one-out\" evaluation, I get an accuracy of "
         << leave_one_out_cross_validation(data, full_set_of_features, data[0].size()) * 100 << "%" << '\n'
         << "Beginning search." << '\n';

    double best_accuracy = -1;                  // Store the final best accuracy.
    vector<int> best_set_of_features;           // Store the final best feature subset found.
    vector<int> current_set_of_features;        // Store the current best feature subset found.

    for (int i = 1; i <= data[0].size() - 1; i++) {
        int feature_to_add_at_this_level = -1;
        double accuracy = 0;                    // Store the current accuracy.
        double best_so_far_accuracy = -1;       // Store the current best accuracy.

        // Iterate over features to determine the best one to add.
        for (int k = 1; k <= data[0].size() - 1; k++) {
            if (find(current_set_of_features.begin(), current_set_of_features.end(), k) == current_set_of_features.end()) {
                // Only consider adding if not already added.
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k);

                // Tell user what features are used in this subset and its accuracy.
                cout << "     Using feature(s) {" << k;
                for (int j = 0; j < current_set_of_features.size(); j++) {
                    cout << ',' << current_set_of_features[j];
                }
                cout << "} accuracy is " << accuracy * 100 << '%' << '\n';

                // Update current best accuracy and feature set if improved.
                if (accuracy > best_so_far_accuracy) {
                    best_so_far_accuracy = accuracy;
                    feature_to_add_at_this_level = k;
                }
            }
        }

        current_set_of_features.insert(current_set_of_features.begin(), feature_to_add_at_this_level);

        // Update the final best accuracy and feature set if improved.
        if (best_so_far_accuracy > best_accuracy) {
            best_accuracy = best_so_far_accuracy;
            best_set_of_features = current_set_of_features;
        }
        else {
            if (i < data[0].size() - 1) {
                cout << "(Warning, Accuracy has decreased! Continuing search in case of local maxima)" << '\n';
            }
        }
        
        // Tell user what the current best subset and accuracy is.
        if (i < data[0].size() - 1) {
            cout << "Feature set {" << current_set_of_features[0];
            for (int k = 1; k < current_set_of_features.size(); k++) {
                cout << ',' << current_set_of_features[k];
            }
            cout << "} was the best, accuracy is " << best_so_far_accuracy * 100 << '%' <<'\n';
        }
    }

    // Tell user what the final best subset and accuracy is.
    cout << "Finished search!! The best feature subset is {" << best_set_of_features[0];
    for (int i = 1; i < best_set_of_features.size(); i++) {
        cout << ',' << best_set_of_features[i];
    }
    cout << "}, which has an accuracy of " << best_accuracy * 100 << '%' <<'\n';

    return;
}

// Backward Selection Algorithm: Removes one feature at a time.
void feature_search_backward(vector<vector<double>> data) {
    // Tell user information about the data.
    cout << "This dataset has " << data[0].size() - 1 << " features (not including the class attribute), with " << data.size() << " instances." << '\n'
         << "Running nearest neighbor with all " << data[0].size() - 1 << " features, using \"leaving-one-out\" evaluation, I get an accuracy of "
         << leave_one_out_cross_validation(data, {}, data[0].size()) * 100 << "%" << '\n'
         << "Beginning search." << '\n';

    double best_accuracy = -1;                  // Store the final best accuracy.
    vector<int> best_set_of_features;           // Store the final best feature subset removed.
    vector<int> current_set_of_features;        // Store the current best feature subset removed.

    for (int i = 1; i <= data[0].size() - 1; i++) {
        int feature_to_add_at_this_level = -1;
        double accuracy = 0;                    // Store the current accuracy.
        double best_so_far_accuracy = -1;       // Store the current best accuracy.

        // Iterate over features to determine the best one to remove.
        for (int k = 1; k <= data[0].size() - 1; k++) {
            if (find(current_set_of_features.begin(), current_set_of_features.end(), k) == current_set_of_features.end()) {
                // Only consider removing if not already removed.
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k);

                // Tell user what features are used in this subset and its accuracy.
                cout << "     Using feature(s) {";
                int j;
                for (j = 1; j <= data[0].size() - 1; j++) {
                    if ((k != j) && (find(current_set_of_features.begin(), current_set_of_features.end(), j) == current_set_of_features.end())) {
                        cout << j; j++; break;
                    }
                }
                while (j <= data[0].size() - 1) {
                    if ((k != j) && (find(current_set_of_features.begin(), current_set_of_features.end(), j) == current_set_of_features.end())) {
                        cout << ',' << j;
                    }
                    j++;
                }
                cout << "} accuracy is " << accuracy * 100 << '%' << '\n';

                // Update current best accuracy and feature set if improved.
                if (accuracy > best_so_far_accuracy) {
                    best_so_far_accuracy = accuracy;
                    feature_to_add_at_this_level = k;
                }
            }
        }

        current_set_of_features.insert(current_set_of_features.begin(), feature_to_add_at_this_level);

        // Update the final best accuracy and feature set if improved.
        if (best_so_far_accuracy > best_accuracy) {
            best_accuracy = best_so_far_accuracy;
            best_set_of_features = current_set_of_features;
        }
        else {
            if (i < data[0].size() - 1) {
                cout << "(Warning, Accuracy has decreased! Continuing search in case of local maxima)" << '\n';
            }
        }
        
        // Tell user what the current best subset and accuracy is.
        if (i < data[0].size() - 1) {
            cout << "Feature set {";
            int k;
            for (k = 1; k <= data[0].size() - 1; k++) {
                if (find(current_set_of_features.begin(), current_set_of_features.end(), k) == current_set_of_features.end()) {
                    cout << k; k++; break;
                }
            }
            while (k <= data[0].size() - 1) {
                if (find(current_set_of_features.begin(), current_set_of_features.end(), k) == current_set_of_features.end()) {
                    cout << ',' << k;
                }
                k++;
            }
            cout << "} was the best, accuracy is " << best_so_far_accuracy * 100 << '%' <<'\n';
        }
    }

    // Tell user what the final best subset and accuracy is.
    cout << "Finished search!! The best feature subset is {";
    int i;
    for (i = 1; i <= data[0].size() - 1; i++) {
        if (find(best_set_of_features.begin(), best_set_of_features.end(), i) == best_set_of_features.end()) {
            cout << i; i++; break;
        }
    }
    while (i <= data[0].size() - 1) {
        if (find(best_set_of_features.begin(), best_set_of_features.end(), i) == best_set_of_features.end()) {
            cout << ',' << i;
        }
        i++;
    }
    cout << "}, which has an accuracy of " << best_accuracy * 100 << '%' <<'\n';

    return;
}

// Evaluates accuracy using Leave-One-Out Cross Validation.
double leave_one_out_cross_validation(vector<vector<double>> data, vector<int> current_set, int feature_to_add) {
    current_set.push_back(feature_to_add);          // Add the selected feature to the current set.

    // Determine whether the current set is what should be added or be removed depending on the search algorithm.
    for (int i = 0; i < data.size(); i++) {
        for (int j = 1; j < data[i].size(); j++) {
            if ((ALGORITHM_TYPE == 1) && (find(current_set.begin(), current_set.end(), j) == current_set.end())) {
                data[i][j] = 0;
            }
            else if ((ALGORITHM_TYPE == 2) && !(find(current_set.begin(), current_set.end(), j) == current_set.end())) {
                data[i][j] = 0;
            }
        }
    }

    int number_correctly_classified = 0;

    for (int i = 0; i < data.size(); i++) {
        vector<double> object_to_classify(data[i].begin()+1, data[i].end());    // Features.
        double label_object_to_classify = data[i][0];                           // Object's label.

        // Variables to track the nearest neighbor.
        double nearest_neighbor_distance = numeric_limits<double>::infinity();
        int nearest_neighbor_location = numeric_limits<int>::infinity();
        double nearest_neighbor_label = -1;

        // Iterate through the dataset to find the nearest neighbor.
        for (int k = 0; k < data.size(); k++) {
            if (k != i) {           // Leave one out.
                double distance = 0.0;

                // Compute Euclidean distance between object_to_classify and current data point.
                for (int j = 1; j < data[k].size(); j++) {
                    distance += pow(object_to_classify[j - 1] - data[k][j], 2);
                }
                distance = sqrt(distance);

                // Update nearest neighbor if a closer one is found.
                if (distance < nearest_neighbor_distance) {
                    nearest_neighbor_distance = distance;
                    nearest_neighbor_location = k;
                    nearest_neighbor_label = data[nearest_neighbor_location][0];
                }
            }
        }

        if (label_object_to_classify == nearest_neighbor_label) {
            number_correctly_classified++;
        }
    }

    return static_cast<double>(number_correctly_classified) / data.size();      // Accuracy.
}