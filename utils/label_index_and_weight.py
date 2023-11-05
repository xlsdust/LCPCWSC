import numpy as np


def get_label_info(dataset):
    if dataset == 'mydata':
        return get_mydata_label_set(), get_mydata_label_weight(), get_mydata_label_prior(), get_mydata_index_to_label()
    elif dataset == 'codata':
        return get_codata_label_set(), get_codata_label_weight(), get_codata_label_prior(), get_codata_index_to_label()


def get_mydata_label_set():
    label_set = "advertising#" \
                "analytics#" \
                "development#" \
                "auto#" \
                "programmer#" \
                "banking#" \
                "cone#" \
                "chain#" \
                "cloud#" \
                "encryption#" \
                "data#" \
                "database#" \
                "domains#" \
                "education#" \
                "email#" \
                "enterprise#" \
                "entertainment#" \
                "events#" \
                "financial#" \
                "games#" \
                "governments#" \
                "internet#" \
                "location#" \
                "mapping#" \
                "marketing#" \
                "medical#" \
                "messaging#" \
                "music#" \
                "news#" \
                "others#" \
                "payments#" \
                "photos#" \
                "Management#" \
                "estate#" \
                "reference#" \
                "science#" \
                "search#" \
                "security#" \
                "shipping#" \
                "social#" \
                "sports#" \
                "stocks#" \
                "storage#" \
                "telephony#" \
                "tools#" \
                "transportation#" \
                "travel#" \
                "video#" \
                "weather#" \
                "commerce"
    return label_set


# "auto#" \
# "encryption#" \
# "location#" \
# 两个数据集的差别
def get_codata_label_set():
    label_set = "advertising#" \
                "analytics#" \
                "development#" \
                "computer#" \
                "banking#" \
                "cone#" \
                "chat#" \
                "cloud#" \
                "data#" \
                "database#" \
                "domains#" \
                "education#" \
                "email#" \
                "enterprise#" \
                "entertainment#" \
                "events#" \
                "sharing#" \
                "financial#" \
                "games#" \
                "governments#" \
                "images#" \
                "internet#" \
                "mapping#" \
                "marketing#" \
                "medical#" \
                "media#" \
                "messaging#" \
                "music#" \
                "news#" \
                "others#" \
                "payments#" \
                "photos#" \
                "Management#" \
                "estate#" \
                "reference#" \
                "science#" \
                "search#" \
                "security#" \
                "shipping#" \
                "social#" \
                "sports#" \
                "stocks#" \
                "storage#" \
                "telephony#" \
                "tools#" \
                "transportation#" \
                "travel#" \
                "video#" \
                "weather#" \
                "commerce"
    return label_set


def get_mydata_index_to_label():
    return {0: 'Advertising', 1: 'Analytics', 2: 'Application Development', 3: 'Auto',
            4: 'Backend',
            5: 'Banking',
            6: 'Bitcoin',
            7: 'Blockchain',
            8: 'Cloud',
            9: 'Cryptocurrency',
            10: 'Data',
            11: 'Database',
            12: 'Domains',
            13: 'Education',
            14: 'Email',
            15: 'Enterprise',
            16: 'Entertainment',
            17: 'Events',
            18: 'Financial',
            19: 'Games',
            20: 'Government',
            21: 'Internet of Things',
            22: 'Location',
            23: 'Mapping',
            24: 'Marketing',
            25: 'Medical',
            26: 'Messaging',
            27: 'Music',
            28: 'News Services',
            29: 'Other',
            30: 'Payments',
            31: 'Photos',
            32: 'Project Management',
            33: 'Real Estate',
            34: 'Reference',
            35: 'Science',
            36: 'Search',
            37: 'Security',
            38: 'Shipping',
            39: 'Social',
            40: 'Sports',
            41: 'Stocks',
            42: 'Storage',
            43: 'Telephony',
            44: 'Tools',
            45: 'Transportation',
            46: 'Travel',
            47: 'Video',
            48: 'Weather',
            49: 'eCommerce'}


def get_codata_index_to_label():
    return {0: 'Advertising',
            1: 'Analytics',
            2: 'Application Development',
            3: 'Backend',
            4: 'Banking',
            5: 'Bitcoin',
            6: 'Chat',
            7: 'Cloud',
            8: 'Data',
            9: 'Database',
            10: 'Domains',
            11: 'Education',
            12: 'Email',
            13: 'Enterprise',
            14: 'Entertainment',
            15: 'Events',
            16: 'File Sharing',
            17: 'Financial',
            18: 'Games',
            19: 'Government',
            20: 'Images',
            21: 'Internet of Things',
            22: 'Mapping',
            23: 'Marketing',
            24: 'Media',
            25: 'Medical',
            26: 'Messaging',
            27: 'Music',
            28: 'News Services',
            29: 'Other',
            30: 'Payments',
            31: 'Photos',
            32: 'Project Management',
            33: 'Real Estate',
            34: 'Reference',
            35: 'Science',
            36: 'Search',
            37: 'Security',
            38: 'Shipping',
            39: 'Social',
            40: 'Sports',
            41: 'Stocks',
            42: 'Storage',
            43: 'Telephony',
            44: 'Tools',
            45: 'Transportation',
            46: 'Travel',
            47: 'Video',
            48: 'Weather',
            49: 'eCommerce'}


def get_mydata_label_weight():
    """
    类别重要性程度，使用类别数量的倒数
    :return:
    """
    return [0.36848,
            0.22198,
            0.23529,
            0.17018,
            0.21458,
            0.34036,
            0.27081,
            0.15242,
            0.34628,
            0.30041,
            0.48391,
            0.22494,
            0.17166,
            0.42027,
            0.47651,
            0.69404,
            0.17906,
            0.18646,
            1.34369,
            0.39956,
            0.5357,
            0.23085,
            0.15982,
            0.63633,
            0.18646,
            0.17758,
            0.89382,
            0.31077,
            0.19238,
            0.18942,
            0.87458,
            0.29597,
            0.24269,
            0.19682,
            0.39808,
            0.5875,
            0.3818,
            0.55198,
            0.27081,
            0.66149,
            0.42323,
            0.18794,
            0.15834,
            0.47799,
            1.17647,
            0.44987,
            0.41139,
            0.41287,
            0.28709,
            0.83907]


def get_mydata_label_prior():
    """
    类别先验概率，使用类别数量/总的训练集数量
    :return:
    """
    return np.array(
        [0.018466, 0.011154, 0.011773, 0.008551, 0.010782, 0.017102, 0.013508, 0.00756, 0.01735, 0.014996, 0.024167,
         0.011154, 0.008427, 0.021068, 0.023795, 0.034825, 0.008799, 0.009171, 0.067419, 0.020077, 0.026769, 0.011402,
         0.007932, 0.031974, 0.009171, 0.008923, 0.044863, 0.015615, 0.009667, 0.009419, 0.043748, 0.014872, 0.012145,
         0.009791, 0.019829, 0.029372, 0.019085, 0.027513, 0.013508, 0.03309, 0.021068, 0.009295, 0.007684, 0.023919,
         0.059115, 0.022432, 0.020573, 0.020696, 0.014376, 0.042013])


def get_codata_label_weight():
    """
    类别重要性程度，使用类别数量的倒数
    :return:
    """
    return [0.36848,
            0.22198,
            0.23529,
            0.17018,
            0.21458,
            0.34036,
            0.27081,
            0.15242,
            0.34628,
            0.30041,
            0.48391,
            0.22494,
            0.17166,
            0.42027,
            0.47651,
            0.69404,
            0.17906,
            0.18646,
            1.34369,
            0.39956,
            0.5357,
            0.23085,
            0.15982,
            0.63633,
            0.18646,
            0.17758,
            0.89382,
            0.31077,
            0.19238,
            0.18942,
            0.87458,
            0.29597,
            0.24269,
            0.19682,
            0.39808,
            0.5875,
            0.3818,
            0.55198,
            0.27081,
            0.66149,
            0.42323,
            0.18794,
            0.15834,
            0.47799,
            1.17647,
            0.44987,
            0.41139,
            0.41287,
            0.28709,
            0.83907]


def get_codata_label_prior():
    """
    类别先验概率，使用类别数量/总的训练集数量
    :return:
    """
    return np.array(
        [0.020497, 0.010535, 0.011222, 0.012252, 0.009619, 0.013054, 0.007443, 0.015802, 0.012939, 0.012367, 0.007672,
         0.019695, 0.02336, 0.037673, 0.008703, 0.009848, 0.007787, 0.06275, 0.01855, 0.027482, 0.007558, 0.010306,
         0.033093, 0.007329, 0.007443, 0.010077, 0.046719, 0.017405, 0.007443, 0.014428, 0.042139, 0.016947, 0.012939,
         0.009848, 0.022902, 0.027253, 0.022902, 0.022902, 0.012367, 0.03962, 0.02084, 0.011909, 0.009046, 0.027482,
         0.070194, 0.019924, 0.022215, 0.022902, 0.012023, 0.042597])
