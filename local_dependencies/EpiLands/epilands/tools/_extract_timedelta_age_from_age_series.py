import pandas as pd
import re


def extract_timedelta_age_from_age_series(age_series: pd.Series) -> pd.Series:
    """
    This function takes a Series of strings that contain the birth and death dates and returns a Series of timedeltas.
    """
    print("Began Generating Ages for inputs: {}".format(age_series.unique()))
    birth_death_pattern = re.compile("B\d\d-\d\d-\d\d\d\d_D\d\d-\d\d-\d\d\d\d")
    age_series = age_series.astype(str, copy=True)
    check__birthdeath_format = age_series.str.contains(birth_death_pattern, regex=True)

    if True in check__birthdeath_format.values:
        print("Generating Birth-death ages to ### days...")
        # get the birth-death formatted ages
        # set index to be the original birth-deaths
        birth_death = pd.Series(
            age_series[check__birthdeath_format.values].unique(),
            index=age_series[check__birthdeath_format.values].unique(),
        )

        # generate lambda functions to 1) select only the birth-death and nothing else, then 2) take that and turn it into a timedelta
        find_birth_death = lambda age: birth_death_pattern.search(age)[0]
        to_timedelta = lambda birthdeath: pd.to_datetime(
            birthdeath[1].replace("D", "")
        ) - pd.to_datetime(birthdeath[0].replace("B", ""))

        # map new days ages into the series as values
        birth_death = birth_death.map(
            lambda a: to_timedelta(find_birth_death(a).split("_"))
        )
        for i in birth_death.index:
            # replace all birth-death with ages in timedelta format
            age_series = age_series.replace(i, str(birth_death[i].days) + " days")
            print(i, "has been converted to", str(birth_death[i].days) + " days")

    # if some are not birth-death formatted
    if False in check__birthdeath_format.values:
        print(
            "NOTICE: not all entries are in the format B\d\d-\d\d-\d\d\d\d_D\d\d-\d\d-\d\d\d\d"
        )
        days_pattern = re.compile("\d?\d?\d ?(day)s?")
        check_days_format = age_series.str.contains(days_pattern, regex=True)
        if True in (check_days_format).values:
            print("Entries below are of the form ###days")
            days = age_series.loc[check_days_format]
            print("found: ", days.unique())
            age_series.loc[check_days_format] = days.map(
                lambda day: str(pd.to_timedelta(day).days) + " days"
            )
            for i, j in zip(days.unique(), age_series.loc[check_days_format].unique()):
                print(i, "has been converted to", j)
        if False in (check_days_format + check__birthdeath_format).values:
            print(
                age_series[
                    [not i for i in check_days_format + check__birthdeath_format]
                ].unique(),
                "are not in the birth-death or days age format. These will be returned as is",
            )

    print("Generating Birth-death ages to ### days: Success!")
    return age_series
