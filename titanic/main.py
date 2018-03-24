import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def preprocess(dataset):
    features = dataset[['SibSp', 'Parch']]

    avg_fare = dataset['Fare'].mean()
    dataset.loc[dataset['Fare'].isnull(), 'Fare'] = avg_fare
    features = pd.concat([features, dataset['Fare']], axis=1)

    avg_age = dataset['Age'].mean()
    dataset.loc[dataset['Age'].isnull(), 'Age'] = avg_age
    features = pd.concat([features, dataset['Age']], axis=1)

    pclass = pd.get_dummies(dataset['Pclass'])
    features = pd.concat([features, pclass], axis=1)

    gender = pd.get_dummies(dataset['Sex'])
    features = pd.concat([features, gender], axis=1)

    title = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title = title.replace(['Lady', 'Countess','Capt', 'Col',
        'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    title = pd.get_dummies(title)
    features = pd.concat([features, title], axis=1)
    print(features.shape)

    return features


def main():
    dataset = pd.read_csv('data/train.csv')

    features = preprocess(dataset)
    labels = dataset['Survived'].astype('int32').as_matrix()

    classifier = RandomForestClassifier()

    experimenting = True
    if experimenting:
        train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.3)
        classifier.fit(train_x, train_y)
        train_pred = classifier.predict(train_x)
        test_pred = classifier.predict(test_x)

        print(classifier.score(train_x, train_y))
        print(classification_report(train_y, train_pred))
        print(classification_report(test_y, test_pred))
    else:
        classifier.fit(features, labels)

        testset = pd.read_csv('data/test.csv')
        test_features = preprocess(testset)

        predicted_result = classifier.predict(test_features)
        result = pd.DataFrame(predicted_result, columns=['Survived'])
        result = pd.concat([testset['PassengerId'], result], axis=1)

        result.to_csv(path_or_buf='data/result.csv', index=False)


if __name__ == "__main__":
    main()
