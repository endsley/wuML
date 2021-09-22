
import wuml

def sort_matrix_rows_by_a_column(X, column_names):
	df = wuml.ensure_DataFrame(X)
	sorted_df = df.sort_values(by=column_names)

	return sorted_df
