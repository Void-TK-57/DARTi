
#import numpy
import numpy as np
#import pandas
import pandas as pd
#import scipy
import scipy.stats as sts
import scipy.special as spc
#import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
#import ternary plot lib
import ternary
#import itertools
import itertools
#improt math
import math

import sys

# main class design
class Design:

    # constructor
    def __init__(self, independent_values, dependent_value, max_min_table = None, confiance_interval = 0.95, degree = 1, use_pure_error = False, use_log = False):
        # try to create the design
        try:
            self.confiance_interval = confiance_interval
            self.independent_values = independent_values
            self.dependent_value = dependent_value
            # set table of max and min values for each factor
            self.max_min_table = max_min_table
            self.degree = degree
            # Calculate Independent values Normalized
            self.coded_independent_values = self.Code_Independent_Values(self.independent_values, self.max_min_table)
            # Calculate The values Expanded (as for each interaction)
            self.expanded_coded_independent_values = self.Expand_Independent_Values(self.coded_independent_values)
            self.number_of_independent_variables = len(list(self.independent_values))
            # Calculate Coefficients of the Regression
            self.coefficients = self.Calculate_Coefficients(self.expanded_coded_independent_values, self.dependent_value)
            self.number_of_parameter = len(list(self.coefficients))
            # Calculate the dependent values for the independet values based on the regression
            self.dependent_value_calculated = self.Calculate_Dependent_Value(self.independent_values)
            self.residuos = self.Calculate_Residuos(self.dependent_value_calculated, self.dependent_value)
            # get the Dependent Value per Level and then calculate it per Level
            self.dependent_value_per_level = self.Calculate_Dependent_Value_Per_Level()
            self.dependent_value_calculated_per_level = self.Calculate_Dependent_Value_Per_Level(True)
            # calculate variance table (ANOVA)
            self.variance_table = self.Calculate_Variance_Table()
            self.estimative_variance = self.variance_table["Média Quadrática"]["Resíduos"] if not use_pure_error else self.variance_table["Média Quadrática"]["Erro Puro"]
            self.covariance_table = self.Calculate_Covariance_Table()
            # Update coefficients to add the standard derivation
            self.Update_Coefficients()
            # Calculate the Table with the Statistical Data
            self.statistics_variance_table = self.Calculate_Statistics_Variance_Table(self.variance_table)
        except Exception as err:
            print("Design Module Error")
            print(err)
            self.valid = False
        else:
            self.valid = True
        if use_log:
            self.log_doe()

    # log method
    def log_doe(self):
        log("Degree :", self.degree)
        log("Degree :", self.degree)
        log("Dependent Value per Level :", self.dependent_value_per_level)
        log("Dependent Values :", self.dependent_value)
        log("Dependent Values Calculated :", self.dependent_value_calculated)
        log("Independent Values :", self.independent_values)
        log("Coded Independent Values :", self.coded_independent_values)
        log("Expanded Coded Independent Values :", self.expanded_coded_independent_values)
        log("Coefficients :", self.coefficients)
        log("Estimative Variance :", self.estimative_variance)
        log("Variance Statistics Table :", self.statistics_variance_table)
        log("Regression Valid :", self.Regression_Valid())
        log("Adjust Valid :", self.Adjust_Valid())
        log("Variance Table :", self.variance_table)
        log("Covariance Table :", self.covariance_table)
        log("Residuos :", self.residuos)

    # method to check if the regression is valid
    def Regression_Valid(self, *args):
        # confiance: per centage of probability
        confiance = self.confiance_interval
        # get degree of freedom of residuos and regression
        degree_freedom_regression = self.variance_table["Graus"]["Regressão"]
        degree_freedom_residuo = self.variance_table["Graus"]["Resíduos"]
        # get F distribution
        F = sts.f.ppf(confiance, degree_freedom_regression, degree_freedom_residuo)
        # get regression ration
        ratio_regression = self.variance_table["Média Quadrática"]["Regressão"]/self.variance_table["Média Quadrática"]["Resíduos"]
        #return if R is higher than the F distribution
        return ratio_regression >= F
    
    # method to calculate if there adjust fault
    def Adjust_Valid(self, *args):
        # confiance: per centage of probability
        confiance = self.confiance_interval
        # get degree of freedom of residuos and regression
        degree_freedom_adjust = self.variance_table["Graus"]["Falta de Ajuste"]
        degree_freedom_pure_error = self.variance_table["Graus"]["Erro Puro"]
        # get F distribution
        F = sts.f.ppf(confiance, degree_freedom_adjust, degree_freedom_pure_error)
        # get adjust ration
        try:
            ratio_adjust = self.variance_table["Média Quadrática"]["Falta de Ajuste"]/self.variance_table["Média Quadrática"]["Erro Puro"]
        except:
            return False
        else:
            #return if R is higher than the F distribution
            return ratio_adjust >= F

    # method to update the coefficients
    def Update_Coefficients(self):
        # get coeficients matrix
        coefficients_matrix = self.coefficients.values
        # get index
        coefficients_index = list( self.coefficients.index )
        # add derivation to it
        coefficients_index.append("Desvio")
        # list of standard derivations
        standard_derivations = list()
        # for each column in covariance table
        for column in list(self.covariance_table):
            # get covariance of column an column (aka the standard derivation) and append to the list
            standard_derivations.append( self.covariance_table[column][column] )
        # convert it to numpy
        standard_derivations = np.array(standard_derivations)
        # get square root
        standard_derivations = np.sqrt(standard_derivations)
        # new matrix
        new_coefficietns_matrix = np.concatenate( [coefficients_matrix, np.array([standard_derivations,]) ] )
        # get interval list
        intervals = self.Calculate_Interval_Values(new_coefficietns_matrix)
        # new matrix
        new_coefficietns_matrix = np.concatenate( [new_coefficietns_matrix, intervals[0].reshape(1, len(intervals[0]) ), intervals[1].reshape(1, len(intervals[1]) ) ] )
        # add derivation to it
        coefficients_index.append("Mínimo Intervalo")
        coefficients_index.append("Máximo Intervalo")

        # create a dataframe
        dataframe = pd.DataFrame(new_coefficietns_matrix, columns = self.coefficients.columns, index = coefficients_index)
        
        # set coefficients to dataframe
        self.coefficients = dataframe

    # method to calculate minimum and maximum interval for the confidence
    def Calculate_Interval_Values(self, independent_values):
        coefficients = independent_values[0, :].flatten()
        standards = independent_values[1, :].flatten()
        # calculat t degree of freedom
        dg = len(self.dependent_value.index) - 2
        # change confiance to 2 tails version for the t distributuion
        confiance_interval = 1.0 - (1.0 - self.confiance_interval)/2.0
        # get t coefficient
        t = sts.t.ppf(confiance_interval, dg)
        # get min and max interval by using the formula: coefficient +- T*std
        min_interval = np.minimum(coefficients - t*standards, coefficients + t*standards)
        max_interval = np.maximum(coefficients - t*standards, coefficients + t*standards)
        # return list of the interval
        return [min_interval, max_interval]
        
    # method to calcualte the covariance table
    def Calculate_Covariance_Table(self):
        # X matrix
        x_matrix = self.expanded_coded_independent_values.values
        # calculate the product by its transpose
        product_matrix = x_matrix.T @ x_matrix
        # calcualte the inverse
        inverse_matrix = np.linalg.inv(product_matrix)
        # multiplicates it by the variance
        covariance = inverse_matrix * self.estimative_variance
        # columns
        columns = list(self.coefficients)
        # index
        index = columns
        # create dataframe
        dataframe = pd.DataFrame(covariance, columns = columns, index = index)
        # return the covariance
        return dataframe

    # method to get the correlation matrix X'X
    def Calculate_Correlation_Matrix(self):
        # X matrix
        x_matrix = self.expanded_coded_independent_values.values
        # calculate the product by its transpose
        product_matrix = x_matrix.T @ x_matrix
        # create dataframe
        dataframe = pd.DataFrame(product_matrix)
        # return dataframe
        return dataframe

    # method to calculate correlation of effect
    def Calculate_Correlation_Effects_Matrix(self):
        # X matrix
        x_matrix = self.expanded_coded_independent_values.values
        # calculate the product by its transpose
        product_matrix = x_matrix.T @ x_matrix
        # calcualte the inverse
        inverse_matrix = np.linalg.inv(product_matrix)
        # create dataframe
        dataframe = pd.DataFrame(inverse_matrix)
        # return dataframe
        return dataframe

    # method to code independent values
    def Code_Independent_Values(self, independent_values, max_min_table):
        # get coded values based on the independt values
        coded_values = independent_values.copy()
        # check if max_min_table is None
        if max_min_table is None:
            # return copy of independent values# for each label
            return coded_values
        # else, for each col
        for col in list(independent_values):
            # get min and max value
            min_value = max_min_table[col][0]
            max_value = max_min_table[col][1]
            # code function
            code = lambda x: (x - ( (max_value + min_value)/2.0) ) / ( (max_value - min_value) / 2.0 )
            # apply to the series
            coded_values[col] = coded_values[col].apply(code)
        # return coded values
        return coded_values

        # method to calculate the coefficients
    
    # method to calculate the coefficients of the regression
    def Calculate_Coefficients(self, expanded_independent_values, dependent_value):
        # get numpy matrix of independent values
        independent_values_matrix = expanded_independent_values.values
        # dependent value
        dependent_value_matrix = dependent_value.values
        # inverse matrix
        inverse_matrix = np.linalg.inv(independent_values_matrix.T @ independent_values_matrix)
        # matrix transpose y
        matrix_transpose_dependent_matrix = independent_values_matrix.T @ dependent_value_matrix
        # calculate coefficients
        coefficients = inverse_matrix @ matrix_transpose_dependent_matrix
        # create a data frame
        pd_dt = pd.DataFrame(coefficients.T, columns = expanded_independent_values.columns, index = ["Coeficientes",])
        # return pd_dt
        return pd_dt

    # method to calculate aproximate value for dependent vlaue based on the coefficients
    def Calculate_Dependent_Value(self, independent_values):
        # get coded values
        coded_values = self.Code_Independent_Values(independent_values, self.max_min_table)
        # expand it
        expanded_values = self.Expand_Independent_Values(coded_values)
        # get matrix of expanded
        expanded_values_matrix = expanded_values.values
        # get coefficients matrix
        coefficients_matrix = self.coefficients.values[0, :].T 
        # get the product of expanded and coefficients
        dependent_values_matrix = expanded_values_matrix @ coefficients_matrix
        # create a dataframe
        pd_dependent_value = pd.DataFrame(np.array(dependent_values_matrix), columns = self.dependent_value.columns)
        # return the dataframe
        return pd_dependent_value

    # method to calculate reisudos
    def Calculate_Residuos(self, calculated_dependent_values, actual_dependent_values):
        # get calculated matrix
        calculated_dependent_values_matrix = calculated_dependent_values.values
        # get actual matrix
        actual_dependent_values_matrix = actual_dependent_values.values
        # get residuos
        residuos_matrix = actual_dependent_values_matrix - calculated_dependent_values_matrix
        # create dataframe
        pd_residuos = pd.DataFrame(residuos_matrix, columns = actual_dependent_values.columns)
        # return pd_residuos
        return pd_residuos

    # method to expand the independent values
    def Expand_Independent_Values(self, coded_values):
        # list of series
        series = []
        # labels
        labels = []
        # get coded values matrix
        coded_values_matrix = coded_values.values
        # for each iteration of variables
        for degree in range(self.degree+1):
            # power coded matrix to the degree
            power_coded = np.power(coded_values_matrix, degree)
            # added it to the series list
            series.append( power_coded )
            # add it to labels
            labels.append(str(degree))
        
        # create as numpy array
        series = np.concatenate(series, axis=1)
        # create a dataframe
        pd_expanded = pd.DataFrame(series, columns = labels)
        # return pd_expanded
        return pd_expanded

    # method to calculate the variance table
    def Calculate_Variance_Table(self):
        #get y values mean
        mean = self.dependent_value.iloc[:, 0].mean()
        # data
        data = [[],[],[],[],[]]
        #indexs
        indexs = ["Regressão", "Resíduos", "Falta de Ajuste", "Erro Puro", "Total"]
        #columns names
        columns = ["Soma Quadrática", "Graus", "Média Quadrática"]
        #quadratic sum lambda
        quadratic_sum = lambda y: np.power( (y - mean) , 2)
        
        #get series of the total quadratic sum
        pd_total = self.dependent_value.iloc[:, 0].copy()
        #get freedom degree
        dg_total = len(pd_total) - 1.0
        #get sum o quadratic sum applied
        pd_total_sum = sum(pd_total.apply(quadratic_sum))
        #add to the total sum
        data[4].append( pd_total_sum)
        #add to data the degrees of freedom
        data[4].append(dg_total)
        #add to data the ratio
        data[4].append(pd_total_sum/dg_total)
        
        #get series of regression
        pd_regression = self.dependent_value_calculated.iloc[:, 0].copy()
        #get sum o quadratic sum applied
        pd_regression_sum = sum(pd_regression.apply(quadratic_sum))
        #add to dict
        data[0].append(pd_regression_sum)
        #get freedom degree
        dg_regression = self.number_of_parameter - 1.0
        #add to data
        data[0].append(dg_regression)
        #add to data
        data[0].append(pd_regression_sum/dg_regression)
        
        #get series of residual
        pd_residual = pd_regression - pd_total
        #get sum
        pd_residual_sum = sum(np.power(pd_residual, 2))
        #add to dict
        data[1].append(pd_residual_sum)
        #get freedom degree
        dg_residual = dg_total - dg_regression
        #add to data
        data[1].append(dg_residual)
        #add to data
        data[1].append(pd_residual_sum/dg_residual)
        
        #get Pure Error and Falta de Ajuste
        # copy dependent value per level table
        dependent_value_per_level = self.dependent_value_per_level.copy()       
        # copy dependent value calculated per level table
        dependent_value_calculated_per_level = self.dependent_value_calculated_per_level.copy()       
        # calculate mean of dependent values per level
        mean_per_level = np.mean(dependent_value_per_level)
        #current level
        level = 0
        #for each per level
        for mean_of_level in mean_per_level:
            #code lambda: (return x - mean squared, if x is a number, else, return 0)
            code = lambda x : np.power( (mean_of_level - x) , 2) if x is not np.NaN else 0
            # apply code on the series column of that level
            dependent_value_per_level.iloc[:, level] = dependent_value_per_level.iloc[:, level].apply(code)
            # apply code on the series column of that level
            dependent_value_calculated_per_level.iloc[:, level] = dependent_value_calculated_per_level.iloc[:, level].apply(code)
            #increase level
            level += 1
            
        #change NaN to 0.0
        dependent_value_per_level.fillna(0.0, inplace = True)
        #get sum of pure error by double sum
        pure_error = np.sum(np.sum(dependent_value_per_level))
        
        #change NaN to 0.0
        dependent_value_calculated_per_level.fillna(0.0, inplace = True)
        #get sum of adjust by double sum
        adjust = np.sum(np.sum(dependent_value_calculated_per_level))
        
        #add to data
        data[3].append(pure_error)
        #get freedom degree (dg_total = n-1, so dg_total - level + 1 = n-1 - m + 1 = n-m )
        dg_pure_error = dg_total - level + 1
        #add to data
        data[3].append(dg_pure_error)
        #get mean error
        mean_error = (pure_error/dg_pure_error) if dg_pure_error != 0 else 0
        #add to data
        data[3].append(mean_error)
        
        #add to data
        data[2].append(adjust)
        #get freedom degree
        dg_adjust = level - self.number_of_parameter
        #add to data
        data[2].append(dg_adjust)
        #get mean error
        mean_adjust = (adjust/dg_adjust) if dg_adjust != 0 else 0
        #add to data
        data[2].append(mean_adjust)

        #create dataframe
        variance_table = pd.DataFrame(data, index = indexs, columns = columns)
        
        #return pd_table
        return variance_table

    # method to calculate variance per sample
    def Calculate_Dependent_Value_Per_Level(self, use_calc_values = False):
        # independent values
        independent_values = self.independent_values
        # dependent values 
        dependent_value = self.dependent_value
        #check if use_calc_values == True
        if use_calc_values:
            # get dependent values by the calculated
            dependent_value = self.dependent_value_calculated

        # get index of duplicated independent values 
        index_of_duplicated_independent_values = independent_values.duplicated(keep = False)
        
        # get index of not duplicated independent_values 
        index_of_unique_independent_values =  -index_of_duplicated_independent_values
        
        #list of series of each level
        list_series_level = []
        
        #level of the series
        level = 0
        
        #First: Create a series with only one value ( y ) for each row not duplicated
        #   -add each series to list
        #second: Create a Series with all the values for the row duplicated
        #   -add each series to list
        #for unique index
        for index, value in dependent_value[dependent_value.columns[0]][index_of_unique_independent_values].iteritems():
            #get series (to have only one value, the unique dependent value of each index)
            series = pd.Series( [value,], index = [index,])
            #change its name to the current level
            series.name = str(level)
            #increase level
            level += 1
            #add series to list
            list_series_level.append(series)
            
        # get data_frame of duplicates dependent values without the duplicates (as a set)
        set_duplicate_independent_values = independent_values[index_of_duplicated_independent_values].drop_duplicates()
        #for each duplicated value of the set of values
        for key, value in set_duplicate_independent_values.iterrows():
            # get bool dataframe where the dependent values are equal to value of set (aka. a bool datafrae with only the duplicates equal to value (other duplicates will be false in the loop))
            bool_dataframe = independent_values[index_of_duplicated_independent_values] == independent_values.iloc[key]
            # apply all() to check each index has all columns values TRUE (aka the row is equal to value)
            index_of_duplicates_of_value = bool_dataframe.apply(lambda x: x.all(), axis = 1)
            # get the rows of the duplicated table indicated by pd_bool_row
            # get the dependent value where the index of duplicates is True
            dependent_value_on_level = dependent_value.where(index_of_duplicates_of_value)
            # get copy o serties
            dependent_value_on_level_series = dependent_value_on_level.copy()
            # change its name
            dependent_value_on_level_series.name = str(level)
            #increase level
            level += 1
            #add it to th list
            list_series_level.append(dependent_value_on_level_series)

       #concatenate each series on the list
        total_dependet_value_per_level = pd.concat(list_series_level, axis = 1, sort=True)
        
        #return the data frame per level
        return total_dependet_value_per_level

    # method to calculate Statistics Variance Table
    def Calculate_Statistics_Variance_Table(self, variance_table):
        #columns names
        columns = ["Variância Explicada", "Máxima Variância Explicada"]
        # index
        index = ["Valor", "Por Centagem"]
        # variance explic.
        variace_explic = variance_table["Soma Quadrática"]["Regressão"]/variance_table["Soma Quadrática"]["Total"]
        # max variance explic
        max_variance_explic = 1.0 - variance_table["Soma Quadrática"]["Erro Puro"]/variance_table["Soma Quadrática"]["Total"]
        # create a dataframe
        dataframe = pd.DataFrame([[variace_explic, max_variance_explic], [variace_explic*100.0, max_variance_explic*100.0]], columns = columns, index = index)
        
        print(dataframe)
        # column names
        columns = list(self.coefficients.columns) + ["Regressão", "Falta de Ajuste"]
        # get matrix
        coefficients_matrix = self.coefficients.values[:, :]
        # boolean array if interval has same signal
        valid = np.logical_not(np.logical_xor(coefficients_matrix[2, :] > 0, coefficients_matrix[3, :] > 0) ) 
        # string list if valid
        string_valid = []
        # change array
        for i in range(len(valid)):
            # check if if valid is true and append string
            if valid[i]:
                string_valid.append("Significativo")
            else:
                string_valid.append("Não Significativo")
        # convert to numpy
        string_valid = np.array(string_valid)
        # extra values
        extra = np.array(["Válida" if self.Regression_Valid() else "Inválida", "Sim" if self.Adjust_Valid() else "Não"])
        # concatenate to valid row
        valid = np.concatenate( (string_valid, extra) )
        # reshape it as 1 row and n columns
        valid = valid.reshape(1, len(valid))
        # create dataframe
        dataframe2 = pd.DataFrame(valid, columns = columns)
        print(dataframe2)
        # final dataframe
        final_dataframe = pd.concat( [dataframe2, dataframe.round(5)], join = 'outer', axis = 1)
        # transpose dataframe
        values = final_dataframe.values.T
        columns = list(final_dataframe.index)
        index = list(final_dataframe.columns)
        # recreate dataframe
        final_dataframe = pd.DataFrame(values, index =index, columns = columns)
        # return dataframe
        return final_dataframe

    # method to plot the design
    def plot(self, sampled_values = True, sample_size = 50.0, original_values = True, residuos = False):
        # ger pyplot figure
        #figure = plt.figure()
        # check there is 2 coordinates 1 for independent and 1 for dependent
        if self.number_of_independent_variables != 1:
            # then cancel plot method
            return

        # check with use sample values
        if sampled_values:
            # get minimum and maximum value independent values
            min_value, max_value = np.min(self.independent_values.values[:, 0]), np.max(self.independent_values.values[:, 0])
            # function to genreate the independent value basebd on the index of the list
            generate_independent = lambda i : (max_value - min_value)*i/sample_size + min_value
            # get sampled independent dataframe
            sampled_independent_list = [ list( [generate_independent(i) for i in range(int(sample_size)) ] ), ]
            # convert to numpy array vector
            sampled_independent_array = np.array(sampled_independent_list).T
            # create the sampled values dataframes
            dataframe_sampled_independent_values = pd.DataFrame(sampled_independent_array, columns = list(self.independent_values))
            # get calculated dependent values dataframe for the sampled independent values
            dataframe_sampled_dependent_values = self.Calculate_Dependent_Value(dataframe_sampled_independent_values)
            # get sampled values as 1D list
            sampled_independent_list = dataframe_sampled_independent_values.values.flatten()
            sampled_dependent_list = dataframe_sampled_dependent_values.values.flatten()
            # plot based on independent and dependent samples
            plt.plot(sampled_independent_list, sampled_dependent_list)

        # check with use of original values
        if original_values:
            # get list of independent values
            independent_values_list = self.independent_values.values.flatten()
            # get list of dependent values
            dependent_values_list = self.dependent_value.values.flatten()
            # plot based on independent and dependent samples
            plt.scatter(independent_values_list, dependent_values_list, c = 'red', marker = '.')


        # show plot
        plt.show()

    # methdo to plot erro
    def error_plot(self, use_original = True, use_calc = True, figure = None):
        # check if figure is None
        if figure is None:
            #create figure
            fig = plt.figure()
            #get axis
            ax = fig.gca()
        else:
            # create axis
            ax = figure.add_subplot(111, )

        # get residuos array
        residuos = self.residuos.values.flatten()
        # get original values
        original = self.dependent_value.values.flatten()
        # calculated
        calculated = self.dependent_value_calculated.values.flatten()
        # ensaios
        ensaios = np.array(list(self.independent_values.index))
        # check if both original and calc true
        if use_original and use_calc:
            ax.scatter(original, calculated, marker = ".")
            ax.set_xlabel("Observados")
            ax.set_ylabel("Previstos")
        elif use_original:
            ax.scatter(original, residuos, marker = ".")
            ax.set_xlabel("Observados")
            ax.set_ylabel("Residuos")
        elif use_calc:
            ax.scatter(calculated, residuos, marker = ".")
            ax.set_xlabel("Previstos")
            ax.set_ylabel("Residuos")
        else:
            ax.scatter(ensaios, residuos, marker = ".")
            ax.set_xlabel("Ensaios")
            ax.set_ylabel("Residuos")
        # add grid
        ax.grid(True)
            
        # if figure is None
        if figure is None:
            # show plot
            plt.show()

# method to do the factorial class desgin
class Factorial_Design(Design):
    
    # constructor
    def __init__(self, independent_values, dependent_value, max_min_table = None, confiance_interval = 0.95, *args, **kwargs):
        # call parent constructor
        super().__init__(independent_values, dependent_value, max_min_table, confiance_interval=confiance_interval, *args, **kwargs)

    # method to expand the independent values
    def Expand_Independent_Values(self, coded_values):
        # list of series
        series = [np.ones( [coded_values.shape[0], 1 ]) , ]
        # labels
        labels = ["0"]
        # get coded values matrix
        coded_values_matrix = coded_values.values
        # for each columns of coded_values
        for column in list(coded_values):
            # get the numpy array of
            series_array = coded_values[column].values
            # reshape it to column vector
            series_array = series_array.reshape( [series_array.shape[0], 1] )
            # added it to the series list
            series.append( series_array )
            # add it to labels
            labels.append(column)
        # create as numpy array
        series = np.concatenate(series, axis=1)
        # create a dataframe
        pd_expanded = pd.DataFrame(series, columns = labels)
        # return pd_expanded
        return pd_expanded

    # method to plot the design
    def plot(self, sampled_values = True, sample_size = 20, original_values = True, residuos = False, surface_plot = True, figure = None):
        #check if number of variables = 2 (else, returns False)
        if self.number_of_independent_variables != 2:
            #return None
            return False

        # check if figure is None
        if figure is None:
            #create figure
            fig = plt.figure()
            #get axis
            ax = fig.gca(projection='3d')
        else:
            # create axis
            ax = figure.add_subplot(111, projection="3d")

        #set x label to the name of the first column of the dataframe
        ax.set_xlabel(self.independent_values.columns[0])
        #set y label to the name of the second column of the dataframe
        ax.set_ylabel(self.independent_values.columns[1])
        #set z label to the name of the second column of the dataframe
        ax.set_zlabel(self.dependent_value.columns[0])

        #get values of first variable
        x1 = self.independent_values.values[:,0]
        #get values of second variable
        x2 = self.independent_values.values[:,1]
        #get y values
        y = self.dependent_value.values[:, 0]

        #check if it should plot original points values
        if original_values:
            #create a scatter plot of the points (x1, x2, y)
            scatter_plot = ax.scatter(x1, x2, y)

        #genereate x1 and x2 samples along the interval of minof x1, x2 and max of x1, x2
        x1 = np.arange(min(x1), max(x1), (max(x1)-min(x1))/sample_size ).reshape(sample_size, 1)
        x2 = np.arange(min(x2), max(x2), (max(x2)-min(x2))/sample_size ).reshape(sample_size, 1)

        #use mesh grid to simulate the cartesian product of x1 and x2, by converting then to column vector form
        #mesh grid
        x1_mesh, x2_mesh = np.meshgrid(x1, x2)

        #shape then again as colmun vectors
        x1_mesh_col = x1_mesh.reshape(len(x1)*len(x2), 1)
        x2_mesh_col = x2_mesh.reshape(len(x1)*len(x2), 1)

        # concatenate them to create a matrix of independent values
        sampled_independent_values_matrix = np.concatenate((x1_mesh_col, x2_mesh_col), 1)

        # create a dataframe to calculate the dependent values
        sampled_independent_values_dataframe = pd.DataFrame(sampled_independent_values_matrix, columns = self.independent_values.columns)

        # calculate the dependent values
        sampled_dependent_value = self.Calculate_Dependent_Value(sampled_independent_values_dataframe)

        # get sampled dependent value matrix and reshape as shape(x1, x2)
        sampled_dependent_value_matrix = sampled_dependent_value.values.reshape(len(x1), len(x2))

        #check if it should be surface plot
        if surface_plot:
            # plot
            surface = ax.plot_surface(x1_mesh, x2_mesh, sampled_dependent_value_matrix, cmap=cm.viridis)
            # set title
            ax.set_title('Superfíce de Resposta')
        else:
            # plot
            surface = ax.contour(x1_mesh, x2_mesh, sampled_dependent_value_matrix, cmap=cm.viridis)
            # set title
            ax.set_title('Superfíce de Resposta')
        # if figure is None
        if figure is None:
            #show plot
            plt.show()

        #return True
        return True

    # method to plot the superfice
    def plot_superfice(self, sampled_values = True, sample_size = 20, figure = None):
        #check if number of variables = 2 (else, returns False)
        if self.number_of_independent_variables != 2:
            #return None
            return False

        #create figure
        #fig = plt.figure()
        # check if figure is None
        if figure is not None:
            #create figure
            ax = figure.add_subplot(111)
        
        #get values of first variable
        x1 = self.independent_values.values[:,0]
        # min value
        x1_min = np.min(x1)
        # min value
        x1_max = np.max(x1)
        #get values of second variable
        x2 = self.independent_values.values[:,1]
        # min value
        x2_min = np.min(x2)
        # min value
        x2_max = np.max(x2)
        #get y values
        y = self.dependent_value.values[:, 0]

        #genereate x1 and x2 samples along the interval of minof x1, x2 and max of x1, x2
        x1 = np.arange(min(x1), max(x1), (max(x1)-min(x1))/sample_size ).reshape(sample_size, 1)
        x2 = np.arange(min(x2), max(x2), (max(x2)-min(x2))/sample_size ).reshape(sample_size, 1)

        #use mesh grid to simulate the cartesian product of x1 and x2, by converting then to column vector form
        #mesh grid
        x1_mesh, x2_mesh = np.meshgrid(x1, x2)

        #shape then again as colmun vectors
        x1_mesh_col = x1_mesh.reshape(len(x1)*len(x2), 1)
        x2_mesh_col = x2_mesh.reshape(len(x1)*len(x2), 1)

        # concatenate them to create a matrix of independent values
        sampled_independent_values_matrix = np.concatenate((x1_mesh_col, x2_mesh_col), 1)

        # create a dataframe to calculate the dependent values
        sampled_independent_values_dataframe = pd.DataFrame(sampled_independent_values_matrix, columns = self.independent_values.columns)

        # calculate the dependent values
        sampled_dependent_value = self.Calculate_Dependent_Value(sampled_independent_values_dataframe)

        # get sampled dependent value matrix and reshape as shape(x1, x2)
        sampled_dependent_value_matrix = sampled_dependent_value.values.reshape(len(x1), len(x2))

        # chekc figure
        if figure is None:
            # create color map
            color_map = plt.imshow(sampled_dependent_value_matrix, extent = [x1_min, x1_max, x2_max, x2_min])
            # set camp
            color_map.set_cmap(cm.viridis)

            #show plot
            plt.show()
        else:
            # create color map
            color_map = ax.imshow(sampled_dependent_value_matrix, extent = [x1_min, x1_max, x2_min, x2_max])
            # set camp
            color_map.set_cmap(cm.viridis)

        #return True
        return True
    

# method to do the factorial class desgin
class Central_Composite_Design(Design):
    
    # constructor
    def __init__(self, independent_values, dependent_value, max_min_table = None, confiance_interval = 0.95):
        # call parent constructor
        super().__init__(independent_values, dependent_value, max_min_table, degree =2, confiance_interval=confiance_interval)

    # method to expand the independent values
    def Expand_Independent_Values(self, coded_values):
        # list of series
        series = [np.ones( [coded_values.shape[0], 1 ]) , ]
        # labels
        labels = ["0"]
        # get coded values matrix
        coded_values_matrix = coded_values.values
        # for each columns of coded_values
        for column in list(coded_values):
            # get the numpy array of
            series_array = coded_values[column].values
            # reshape it to column vector
            series_array = series_array.reshape( [series_array.shape[0], 1] )
            # added it to the series list
            series.append( series_array )
            # append the square series array
            series.append( np.power( series_array, 2) )
            # add it to labels
            labels.append(column)
            # add it the square
            labels.append(column+"^2")
        
        # for each column combination
        for column_combination in itertools.combinations(list(coded_values), 2):
            # get the first and second array of
            first_series_array = coded_values[column_combination[0]].values
            second_series_array = coded_values[column_combination[1]].values
            # reshape it to column vector
            first_series_array = first_series_array.reshape( [first_series_array.shape[0], 1] )
            second_series_array = second_series_array.reshape( [second_series_array.shape[0], 1] )
            # multiply first by second
            product_series_array = first_series_array * second_series_array
            # added it to the series list
            series.append( product_series_array )
            # add it to labels
            labels.append(column_combination[0] + column_combination[1])    

        # create as numpy array
        series = np.concatenate(series, axis=1)
        # create a dataframe
        pd_expanded = pd.DataFrame(series, columns = labels)
        # return pd_expanded
        return pd_expanded

    # method to plot the design
    def plot(self, sampled_values = True, sample_size = 20, original_values = True, residuos = False, surface_plot = True, figure = None):
        #check if number of variables = 2 (else, returns False)
        if self.number_of_independent_variables != 2:
            #return None
            return False

        # check if figure is None
        if figure is None:
            #create figure
            fig = plt.figure()
            #get axis
            ax = fig.gca(projection='3d')
        else:
            # create axis
            ax = figure.add_subplot(111, projection="3d")

        #set x label to the name of the first column of the dataframe
        ax.set_xlabel(self.independent_values.columns[0])
        #set y label to the name of the second column of the dataframe
        ax.set_ylabel(self.independent_values.columns[1])
        #set z label to the name of the second column of the dataframe
        ax.set_zlabel(self.dependent_value.columns[0])

        #get values of first variable
        x1 = self.independent_values.values[:,0]
        #get values of second variable
        x2 = self.independent_values.values[:,1]
        #get y values
        y = self.dependent_value.values[:, 0]

        #check if it should plot original points values
        if original_values:
            #create a scatter plot of the points (x1, x2, y)
            scatter_plot = ax.scatter(x1, x2, y)

        #genereate x1 and x2 samples along the interval of minof x1, x2 and max of x1, x2
        x1 = np.arange(min(x1), max(x1), (max(x1)-min(x1))/sample_size ).reshape(sample_size, 1)
        x2 = np.arange(min(x2), max(x2), (max(x2)-min(x2))/sample_size ).reshape(sample_size, 1)

        #use mesh grid to simulate the cartesian product of x1 and x2, by converting then to column vector form
        #mesh grid
        x1_mesh, x2_mesh = np.meshgrid(x1, x2)

        #shape then again as colmun vectors
        x1_mesh_col = x1_mesh.reshape(len(x1)*len(x2), 1)
        x2_mesh_col = x2_mesh.reshape(len(x1)*len(x2), 1)

        # concatenate them to create a matrix of independent values
        sampled_independent_values_matrix = np.concatenate((x1_mesh_col, x2_mesh_col), 1)

        # create a dataframe to calculate the dependent values
        sampled_independent_values_dataframe = pd.DataFrame(sampled_independent_values_matrix, columns = self.independent_values.columns)

        # calculate the dependent values
        sampled_dependent_value = self.Calculate_Dependent_Value(sampled_independent_values_dataframe)

        # get sampled dependent value matrix and reshape as shape(x1, x2)
        sampled_dependent_value_matrix = sampled_dependent_value.values.reshape(len(x1), len(x2))

        #check if it should be surface plot
        if surface_plot:
            # plot suface
            surface = ax.plot_surface(x1_mesh, x2_mesh, sampled_dependent_value_matrix, cmap=cm.viridis)
            # set title
            ax.set_title('Superfíce de Resposta')
        else:
            surface = ax.contour(x1_mesh, x2_mesh, sampled_dependent_value_matrix, cmap=cm.viridis)
            # set title
            ax.set_title('Superfíce de Resposta')
        # if figure is None
        if figure is None:
            #show plot
            plt.show()

        #return True
        return True
    
    # method to plot the superfice
    def plot_superfice(self, sampled_values = True, sample_size = 20, figure = None):
        #check if number of variables = 2 (else, returns False)
        if self.number_of_independent_variables != 2:
            #return None
            return False

        #create figure
        #fig = plt.figure()
        # check if figure is None
        if figure is not None:
            #create figure
            ax = figure.add_subplot(111)
        
        #get values of first variable
        x1 = self.independent_values.values[:,0]
        # min value
        x1_min = np.min(x1)
        # min value
        x1_max = np.max(x1)
        #get values of second variable
        x2 = self.independent_values.values[:,1]
        # min value
        x2_min = np.min(x2)
        # min value
        x2_max = np.max(x2)
        #get y values
        y = self.dependent_value.values[:, 0]

        #genereate x1 and x2 samples along the interval of minof x1, x2 and max of x1, x2
        x1 = np.arange(min(x1), max(x1), (max(x1)-min(x1))/sample_size ).reshape(sample_size, 1)
        x2 = np.arange(min(x2), max(x2), (max(x2)-min(x2))/sample_size ).reshape(sample_size, 1)

        #use mesh grid to simulate the cartesian product of x1 and x2, by converting then to column vector form
        #mesh grid
        x1_mesh, x2_mesh = np.meshgrid(x1, x2)

        #shape then again as colmun vectors
        x1_mesh_col = x1_mesh.reshape(len(x1)*len(x2), 1)
        x2_mesh_col = x2_mesh.reshape(len(x1)*len(x2), 1)

        # concatenate them to create a matrix of independent values
        sampled_independent_values_matrix = np.concatenate((x1_mesh_col, x2_mesh_col), 1)

        # create a dataframe to calculate the dependent values
        sampled_independent_values_dataframe = pd.DataFrame(sampled_independent_values_matrix, columns = self.independent_values.columns)

        # calculate the dependent values
        sampled_dependent_value = self.Calculate_Dependent_Value(sampled_independent_values_dataframe)

        # get sampled dependent value matrix and reshape as shape(x1, x2)
        sampled_dependent_value_matrix = sampled_dependent_value.values.reshape(len(x1), len(x2))

        # chekc figure
        if figure is None:
            # create color map
            color_map = plt.imshow(sampled_dependent_value_matrix, extent = [x1_min, x1_max, x2_min, x2_max])
            # set camp
            color_map.set_cmap(cm.viridis)

            #show plot
            plt.show()
        else:
            # create color map
            color_map = ax.imshow(sampled_dependent_value_matrix, extent = [x1_min, x1_max, x2_min, x2_max])
            # set camp
            color_map.set_cmap(cm.viridis)

        #return True
        return True

# method to do a centroid simples
class Mixture_Design(Design):

    # constructor
    def __init__(self, independent_values, dependent_value, max_min_table = None, degree = None, confiance_interval = 0.95):
        # check if degree is None or is higher than the number of variables
        if degree is None or degree > len(list(independent_values)) :
            # set degree to the number of independent_values
            degree = len(list(independent_values))

        # call parent constructor
        super().__init__(independent_values, dependent_value, max_min_table, degree = degree, confiance_interval=confiance_interval)

    # method to expand the independent values
    def Expand_Independent_Values(self, coded_values):
        # values list
        values = []
        # columns list
        columns = []
        # for each interation
        for i in range(self.degree):
            # for each column combination
            for column_combination in itertools.combinations(list(coded_values), i+1):
                # get list of labels
                column_combination_copy = list(column_combination)
                # get list of labels
                labels = list(column_combination)
                # replace column label for the array
                # for each index of col
                for index in range(len(column_combination_copy)):
                    # replace it for the array value
                    column_combination_copy[index] = coded_values[column_combination_copy[index]].values
                # separator for the label
                separator = ' '
                # join the labels
                label = separator.join(labels)
                # add label to columns list
                columns.append(label)
                # convert column combination to matrix
                product_matrix = np.array(column_combination_copy).T
                # product
                product_array = np.prod(product_matrix, axis = 1)
                # add it to values list
                values.append(product_array)
        # convert values to matrix
        values = np.array(values).T
        # creae dataframe
        dataframe = pd.DataFrame(values, columns = columns)
        # return dataframe
        return dataframe 
        
    # method to plot the design
    def plot(self, scale = 30, figure = None):
        #check if number of variables = 3 (else, returns False)
        if self.number_of_independent_variables != 3:
            #return None
            return False

        # check if figure is None
        if figure is None:
            #create figure
            fig, tax = ternary.figure(scale=scale)
        else:
            # create axis
            ax = figure.add_subplot(111)
            #create a figure and a ternary axe (tax)
            tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale, permutation=None)

        # method to rotate
        def rotate(f, x, angle):
            #rotate x values
            rotate_x = x 
            #rotate values
            rotate_x[0], rotate_x[1], rotate_x[2] = rotate_x[(0 + angle) % 3], rotate_x[(1 + angle) % 3], rotate_x[(2 + angle) % 3]
            # return the f function apppling rotate x instead
            return f(x)

        # method to calculate the 
        def calculate(x):
            # convert to numpy array
            x = np.array(x)
            # reshape x
            x = x.reshape(1, len(x))
            # create a dataframe
            x_dataframe = pd.DataFrame(x, columns = self.independent_values.columns)
            # expand it
            expanded_values = self.Expand_Independent_Values(x_dataframe)
            # get matrix of expanded
            expanded_values_matrix = expanded_values.values
            # get coefficients matrix
            coefficients_matrix = self.coefficients.values[0, :].T 
            # get the product of expanded and coefficients
            dependent_values_matrix = expanded_values_matrix @ coefficients_matrix
            # create a dataframe
            return dependent_values_matrix[0]
        
        #create heatmap based on the self.calculate method (which calculate the value of y based on the B values calculated and the x values passed)
        #tax.heatmapf(self.calculate, boundary=True, style="triangular", cmap=cm.plasma)
        tax.heatmapf(calculate, boundary=True, style="triangular", cmap=cm.winter)
        # set boundary
        tax.boundary(linewidth=1.0)
        # set title
        tax.set_title("Gráfico")
        #set axis label
        tax.left_axis_label(self.independent_values.columns[0], fontsize=12)
        tax.right_axis_label(self.independent_values.columns[1], fontsize=12)
        tax.bottom_axis_label(self.independent_values.columns[2], fontsize=12)
        # check if figure is None
        if figure is None:
            #show plot
            tax.show()
        #return True
        return True
    
    # method to plot the design
    def plot_superfice(self, scale = 30, figure = None):
        #check if number of variables = 3 (else, returns False)
        if self.number_of_independent_variables != 3:
            #return None
            return False

        # check if figure is None
        if figure is None:
            #create figure
            fig, tax = ternary.figure(scale=scale)
        else:
            # create axis
            ax = figure.add_subplot(111)
            #create a figure and a ternary axe (tax)
            tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale, permutation=None)

        # method to rotate
        def rotate(f, x, angle):
            #rotate x values
            rotate_x = x 
            #rotate values
            rotate_x[0], rotate_x[1], rotate_x[2] = rotate_x[(0 + angle) % 3], rotate_x[(1 + angle) % 3], rotate_x[(2 + angle) % 3]
            # return the f function apppling rotate x instead
            return f(x)

        # method to calculate the 
        def calculate(x):
            # convert to numpy array
            x = np.array(x)
            # reshape x
            x = x.reshape(1, len(x))
            # create a dataframe
            x_dataframe = pd.DataFrame(x, columns = self.independent_values.columns)
            # expand it
            expanded_values = self.Expand_Independent_Values(x_dataframe)
            # get matrix of expanded
            expanded_values_matrix = expanded_values.values
            # get coefficients matrix
            coefficients_matrix = self.coefficients.values[0, :].T 
            # get the product of expanded and coefficients
            dependent_values_matrix = expanded_values_matrix @ coefficients_matrix
            # create a dataframe
            return dependent_values_matrix[0]
        
        #create heatmap based on the self.calculate method (which calculate the value of y based on the B values calculated and the x values passed)
        #tax.heatmapf(self.calculate, boundary=True, style="triangular", cmap=cm.plasma)
        tax.heatmapf(calculate, boundary=True, style="triangular", cmap=cm.winter)
        # set boundary
        tax.boundary(linewidth=1.0)
        # set title
        tax.set_title("Gráfico")
        #set axis label
        tax.left_axis_label(self.independent_values.columns[0], fontsize=12)
        tax.right_axis_label(self.independent_values.columns[1], fontsize=12)
        tax.bottom_axis_label(self.independent_values.columns[2], fontsize=12)
        # check if figure is None
        if figure is None:
            #show plot
            tax.show()
        #return True
        return True

    # method to create matrix
    def centroid_simplex(factors):
        # create int of factors
        factors = int(factors)
        # check if factors is 
        if factors < 2: 
            factors = 2
        # final list
        final = []
        # indexs
        indexs = np.array(list(range(factors)))
        # for each degree
        for dg in range(1, factors+1):
            # for each combination of index
            for comb in itertools.combinations(indexs, dg):
                # copy indexs
                row = np.zeros(factors)
                # for index in combination
                for index in comb:
                    # chande copy
                    row[index] = 1
                row = row / np.sum(row)
                # append
                final.append(row)
        # create array
        final = np.array(final)
        # return matrix
        return final

    # method to create matrix
    def lattice_simplex(factors, q = 1):
        # create int of factors
        factors = int(factors)
        q = int(q)
        # check if factors is 
        if factors < 2: 
            factors = 2
        # check if q is lower than 1
        if q < 1:
            q = 1
        if q > factors:
            q = factors
        # final list
        final = []
        # first matrix
        generator = np.zeros(factors)
        # set first element to 1
        generator[0] = 1
        # permutate
        for perm in itertools.permutations(generator, factors):
            # add to final
            final.append(perm)
        # check if q is 2
        if q == 2:
            generator = np.zeros(factors)
            # set first element to 1
            generator[0] = .5
            generator[1] = .5
            # permutate
            for perm in itertools.permutations(generator, factors):
                # add to final
                final.append(perm)
        elif q == 3:
            generator = np.zeros(factors)
            # set first element to 1
            generator[0] = 1.0/3.0
            generator[1] = 2.0/3.0
            # permutate
            for perm in itertools.permutations(generator, factors):
                # add to final
                final.append(perm)
            generator = np.zeros(factors)
            # set first element to 1
            generator[0] = 1.0/3.0
            generator[1] = 1.0/3.0
            generator[2] = 1.0/3.0
            # permutate
            for perm in itertools.permutations(generator, factors):
                # add to final
                final.append(perm)
        # create dataframe
        dataframe = pd.DataFrame(final).drop_duplicates()
        # return matrix of the dataframe
        return dataframe.values


def log(message, obj):
    print("========================")
    print(message)
    print(obj)
    
def factorial_test():
    x =[[45,  90],
        [55,  90],
        [45, 110],
        [55, 110],
        [50, 100],
        [50, 100],
        [50, 100]]
    
    y =[[69,],
        [59,],
        [78,],
        [67,],
        [68,],
        [66,],
        [69,]]

    m =[[45.0,  90.0],
        [55.0, 110.0]]

    x =[[30, 115],
        [40, 115],
        [30, 135],
        [40, 135],
        [35, 125],
        [35, 125],
        [35, 125],
        [27.92895, 125],
        [35, 139.1421],
        [42.07105, 125],
        [35, 110.8579]]
    
    y =[[86,],
        [85,],
        [78,],
        [84,],
        [90,],
        [88,],
        [89,],
        [81,],
        [80,],
        [86,],
        [87,]]

    m =[[30.0, 115.0],
        [40.0, 135.0]]

    # create dataframes
    pd_x = pd.DataFrame(x, columns = ["A", "B"])
    pd_y = pd.DataFrame(y, columns = ["Y",])
    pd_m = pd.DataFrame(m, columns = ["A", "B"])

    dataframe = pd.concat([pd_x, pd_y], axis = 1)
    dataframe.to_csv("ensaios.csv")
    pd_m.to_csv("maxmin.csv")

    doe = Factorial_Design(pd_x, pd_y, pd_m)

    log("Covariance Table :", doe.covariance_table)
    log("Degree :", doe.degree)
    log("Dependent Value per Level :", doe.dependent_value_per_level)
    log("Dependent Values :", doe.dependent_value)
    log("Dependent Values Calculated :", doe.dependent_value_calculated)
    log("Independent Values :", doe.independent_values)
    log("Coded Independent Values :", doe.coded_independent_values)
    log("Expanded Coded Independent Values :", doe.expanded_coded_independent_values)
    log("Variance Table :", doe.variance_table)
    log("Variance Statistics Table :", doe.statistics_variance_table)
    log("Estimative Variance :", doe.estimative_variance)
    log("Coefficients :", doe.coefficients)
    log("Regression Valid :", doe.Regression_Valid())
    log("Adjust Valid :", doe.Adjust_Valid())
    doe.plot()
    doe.plot(surface_plot = False)
    doe.plot_superfice()

def regression_test():
    x =[[30,],
        [30,],
        [35,],
        [35,],
        [40,],
        [40,],
        [45,],
        [45,],
        [50,],
        [50,],
        [55,],
        [55,],
        [60,],
        [60,],
        [65,],
        [65,],
        [70,],
        [70,]]
    
    y =[[20,],
        [24,],
        [40,],
        [43,],
        [57,],
        [60,],
        [70,],
        [72,],
        [77,],
        [80,],
        [86,],
        [89,],
        [88,],
        [91,],
        [89,],
        [86,],
        [84,],
        [80,],]

    m =[[-1.0,],
        [+1.0,]]

    x1=[[40,],
        [45,],
        [50,],
        [55,],
        [60,]]
    
    y1=[[60,],
        [70,],
        [77,],
        [86,],
        [91,]]

    # create dataframes
    pd_x = pd.DataFrame(x1, columns = ["X",])
    pd_y = pd.DataFrame(y1, columns = ["Y",])
    pd_m = pd.DataFrame(m, columns = ["X",])

    doe = Design(pd_x, pd_y, pd_m, 0.95, 1, use_log = True)

    doe.plot()
    doe.error_plot(False, False)
    doe.error_plot(True, False)
    doe.error_plot(False, True)
    doe.error_plot(True, True)
    
def central_composite_test():
    x =[[30, 115],
        [40, 115],
        [30, 135],
        [40, 135],
        [35, 125],
        [35, 125],
        [35, 125],
        [27.92895, 125],
        [35, 139.1421],
        [42.07105, 125],
        [35, 110.8579]]
    
    y =[[86,],
        [85,],
        [78,],
        [84,],
        [90,],
        [88,],
        [89,],
        [81,],
        [80,],
        [86,],
        [87,]]

    m =[[30.0, 115.0],
        [40.0, 135.0]]

    x1 =[[45,  90],
        [55,  90],
        [45, 110],
        [55, 110],
        [50, 100],
        [50, 100],
        [50, 100]]
    
    y1 =[[69,],
        [59,],
        [78,],
        [67,],
        [68,],
        [66,],
        [69,]]

    m =[[45.0,  90.0],
        [55.0, 110.0]]

    # create dataframes
    pd_x = pd.DataFrame(x, columns = ["A", "B"])
    pd_y = pd.DataFrame(y, columns = ["Y",])
    pd_m = pd.DataFrame(m, columns = ["A", "B"])

    doe = Central_Composite_Design(pd_x, pd_y, pd_m)

    log("Covariance Table :", doe.covariance_table)
    log("Degree :", doe.degree)
    log("Dependent Value per Level :", doe.dependent_value_per_level)
    log("Dependent Values :", doe.dependent_value)
    log("Dependent Values Calculated :", doe.dependent_value_calculated)
    log("Independent Values :", doe.independent_values)
    log("Coded Independent Values :", doe.coded_independent_values)
    log("Expanded Coded Independent Values :", doe.expanded_coded_independent_values)
    log("Variance Table :", doe.variance_table)
    log("Variance Statistics Table :", doe.statistics_variance_table)
    log("Estimative Variance :", doe.estimative_variance)
    log("Coefficients :", doe.coefficients)
    log("Regression Valid :", doe.Regression_Valid())
    log("Adjust Valid :", doe.Adjust_Valid())
    doe.plot()
    doe.plot(surface_plot = False)
    doe.plot_superfice()

def mixture_test():
    x2 =[[1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [.5, .5, 0],
        [.5, .5, 0],
        [.5, .5, 0],
        [.5, 0, .5],
        [.5, 0, .5],
        [.5, 0, .5],
        [0, .5, .5],
        [0, .5, .5],
        [0, .5, .5],
        [.33, .33, .33],
        [.33, .33, .33],
        [.66, .17, .17],
        [.17, .66, .17],
        [.17, .17, .66]]

    x1 =[[1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [.5, .5, 0],
        [.5, .5, 0],
        [.5, .5, 0],
        [.5, 0, .5],
        [.5, 0, .5],
        [.5, 0, .5],
        [0, .5, .5],
        [0, .5, .5],
        [0, .5, .5],
        [.333, .333, .333],
        [.333, .333, .333]]

    x =[[1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [.5, .5, 0],
        [.5, .5, 0],
        [.5, .5, 0],
        [.5, 0, .5],
        [.5, 0, .5],
        [.5, 0, .5],
        [0, .5, .5],
        [0, .5, .5],
        [0, .5, .5]]

    y2 =[[3.20,],
        [3.00,],
        [0.50,],
        [0.40,],
        [0.40,],
        [0.30,],
        [1.90,],
        [1.20,],
        [2.00,],
        [3.90,],
        [4.40,],
        [4.10,],
        [0.30,],
        [0.30,],
        [0.20,],
        [3.40,],
        [3.60,],
        [4.00,],
        [1.60,],
        [1.80,]]

    y1 =[[3.20,],
        [3.00,],
        [0.50,],
        [0.40,],
        [0.40,],
        [0.30,],
        [1.90,],
        [1.20,],
        [2.00,],
        [3.90,],
        [4.40,],
        [4.10,],
        [0.30,],
        [0.30,],
        [0.20,],
        [3.40,],
        [3.60,]]

    y =[[3.20,],
        [3.00,],
        [0.50,],
        [0.40,],
        [0.40,],
        [0.30,],
        [1.90,],
        [1.20,],
        [2.00,],
        [3.90,],
        [4.40,],
        [4.10,],
        [0.30,],
        [0.30,],
        [0.20,]]
    
    # create dataframes
    pd_x = pd.DataFrame(x1, columns = ["A", "B", "C"])
    pd_y = pd.DataFrame(y1, columns = ["Y",])
    pd_m = None

    doe = Mixture_Design(pd_x, pd_y, degree= 3)

    
    log("Degree :", doe.degree)
    log("Dependent Value per Level :", doe.dependent_value_per_level)
    log("Dependent Values :", doe.dependent_value)
    log("Dependent Values Calculated :", doe.dependent_value_calculated)
    log("Residuos :", doe.residuos)
    log("Independent Values :", doe.independent_values)
    log("Coded Independent Values :", doe.coded_independent_values)
    log("Expanded Coded Independent Values :", doe.expanded_coded_independent_values)
    log("Coefficients :", doe.coefficients)
    log("Estimative Variance :", doe.estimative_variance)
    log("Variance Statistics Table :", doe.statistics_variance_table)
    log("Regression Valid :", doe.Regression_Valid())
    log("Adjust Valid :", doe.Adjust_Valid())
    log("Variance Table :", doe.variance_table)
    log("Covariance Table :", doe.covariance_table)
    doe.plot()
    doe.error_plot(False, False)
    doe.error_plot(True, False)
    doe.error_plot(False, True)
    doe.error_plot(True, True)

if __name__ == "__main__":
    if sys.argv[1] == "regression":
        regression_test()
    elif sys.argv[1] == "factorial":
        factorial_test()
    elif sys.argv[1] == "central_composite":
        central_composite_test()  
    elif sys.argv[1] == "mixture":
        mixture_test()  
    else:
        print("Invalid Parameter")
        