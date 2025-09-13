def scoring_prompt(explanation, inference):
    return f"""You are a reasoning system that analyzes the likelihood of complex events given information about hypothetical scenarios. 

You are given a description of a fictional scenario and a hypothesis about that scenario that may or may not be true. Given the situation, you will first score the likelihood that this hypothesis is true, on a scale from 0 to 10, using the following rubric as guidance:

0 (virtually impossible): Essentially no way the hypothesis could possibly be true, given the evidence. Less likely than being struck by lightning.
1 (unlikely): The hypothesis is unlikely, but definitely not impossible.
2 (possible): The hypothesis could be true given the evidence, but there is better chance that it is false. Less likely than drawing a card of the suit of clubs from a standard card deck.
3 (reasonable chance): You would not be more than mildly surprised that the hypothesis is true. About one thirds chance.
4 (a bit less than even-odds): Slightly below fifty-fifty probability. You would not bet more than a small sum that the hypothesis is false.
5 (fifty-fifty): Given the information about the situation, there is approximately equal chance that the hypothesis is true vs. the hypothesis is false. As likely as a fair coin landing on heads.
6 (a bit more than even-odds): Slightly above fifty-fifty probability. You would not bet more than a small sum that the hypothesis is true.
7 (probable): Likely, but you would still not be overly surprised if the hypothesis turned out to be false.
8 (quite likely): About as likely as \*not\* rolling a “2” with a six-sided die.
9 (extremely likely): Quite certain. You would bet a large amount of money on the hypothesis being true.
10 (practically certain): You cannot imagine a scenario in which the hypothesis is not true, given the situational evidence.

Label your initial prediction with (0), and label your updated predictions with the evidence number it corresponds to. Write your enumerated explanations and probability scores, and nothing else.

Here is a first example:

ORIGINAL DESCRIPTION: There were puddles in the street and dark clouds hung overhead. The Mississippi flag was visible on a nearby car.

HYPOTHESIS: A tornado rolled through a town in Mississippi.

EXPLANATION: It is more likely this was just a regular rainstorm than a tornado. While it might be in Mississippi, there's not enough evidence to claim a tornado occurred.

SCORE: 1

Here is a second example:

ORIGINAL DESCRIPTION: There is a large crowd of people gathered before a lit-up stage at night.

HYPOTHESIS: The band Blur performed at Coachella 2024.

EXPLANATION: While a big nighttime show with a large crowd could describe many concerts or festivals, there is no direct evidence that this specific event is Coachella or that the band on stage is Blur.

SCORE: 2

That is the end of the examples. Now, it’s time for you to assign probabilities to a new fictional scenario:

ORIGINAL DESCRIPTION: {explanation}

HYPOTHESIS: {inference}

EXPLANATION:
"""