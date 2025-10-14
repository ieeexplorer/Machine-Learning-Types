from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample data: Study hours -> Exam score
study_data = {
    'study_hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'exam_score': [45, 52, 58, 65, 70, 74, 78, 82, 85, 88, 90, 92, 94, 95, 96]
}

study_df = pd.DataFrame(study_data)
X_study = study_df[['study_hours']]
y_study = study_df['exam_score']

# Train regression model
reg_model = LinearRegression()
reg_model.fit(X_study, y_study)  # LEARNING: study_hours -> exam_score mapping

# Make predictions
y_pred_reg = reg_model.predict(X_study)

# Evaluate
mse = mean_squared_error(y_study, y_pred_reg)
r2 = r2_score(y_study, y_pred_reg)

print(f"\nRegression Results:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
print(f"Equation: Score = {reg_model.intercept_:.2f} + {reg_model.coef_[0]:.2f} * Study_Hours")

# Predict for new student
new_study_hours = [[8.5]]
predicted_score = reg_model.predict(new_study_hours)
print(f"Predicted score for 8.5 study hours: {predicted_score[0]:.1f}")
