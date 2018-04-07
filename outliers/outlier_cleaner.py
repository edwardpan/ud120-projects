#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    import numpy as np
    # print net_worths
    print np.shape(predictions)

    sub = []
    for i, (p, n) in enumerate(zip(predictions, net_worths)):
        sub.append((i, abs(p[0] - n[0])))
    sub.sort(lambda x, y: cmp(x[1], y[1]), reverse=True)
    # print(sub)
    new_ages = []
    new_net_worths = []
    errors = []
    error_sub = sub[:10]
    # print(len(error_sub))
    sub = sub[9:]
    # print(len(sub))
    for i, _ in sub:
        # print ages[i][0]
        new_ages.append(ages[i][0])
        new_net_worths.append(net_worths[i][0])
    for i, _ in error_sub:
        errors.append(net_worths[i][0])
    # print new_ages
    # print new_net_worths
    # print errors

    cleaned_data = (new_ages, new_net_worths, errors)
    return cleaned_data

