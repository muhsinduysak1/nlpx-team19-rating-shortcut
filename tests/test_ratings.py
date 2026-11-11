"""Unit tests for ratings.py  (run: python -m pytest -q tests  OR  python tests/test_ratings.py)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ratings import find_ratings, has_rating, scrub, invert_x10, last_x10_score, clean_html

POSITIVE = {
    "Great fun. 7/10": "x10_slash",
    "I'd say 7 / 10 overall.": "x10_slash",
    "Solid 7.5/10.": "x10_slash",
    "My rating: 8/10": "x10_slash",
    "Easily 10/10!": "x10_slash",
    "I give it 7 out of 10.": "x10_outof",
    "seven out of ten": "x10_outof",
    "8 out of 10 stars": "x10_outof",
    "Overall 3/5": "x5_slash",
    "3.5/5 stars": "x5_slash",
    "3 out of 5": "x5_outof",
    "three out of five stars.": "x5_outof",
    "2 1/2 out of 4 stars": "x5_outof",
    "***/****": "star_symbols",
    "** 1/2 out of ****": "star_symbols",
    "I'd give this 3 stars.": "digit_stars",
    "a 4-star movie": "digit_stars",
    "I give it two stars.": "verb_word_stars",
    "I would give it a 7.": "verb_number",
    "I rated this film a 3": "verb_number",
    "Rating: 7": "label_number",
    "Grade: B+": "letter_grade",
    "I give it a C-.": "letter_grade",
    "Verdict: 85/100": "x100_slash",
    "Rating: 0 out of 100.": "x100_slash",
    "worst films ever made...1/10": "x10_slash",
    "look no further.9/10": "x10_slash",
    "a classic.10 out of 10": "x10_outof",
    "4.3 (out of 10)": "x10_outof",
    "My Rating: 8 stars out of ten.": "x10_outof",
    "FIVE stars [out of five].": "x5_outof",
    "** (out of four)": "star_symbols",
    "**** out of 5": "star_symbols",
    "The only reason why I even give it 3 stars is because": "digit_stars",
    "I give this series 8 stars.": "digit_stars",
    "Rating 9": "label_number",
}
NEGATIVE = [
    "It came out after 9/11 and it shows.",
    "Open 24/7, like this plot.",
    "The first 1/2 hour is slow.",
    "It is one of the top 10 films of the year.",
    "Released on 3/10/2005 in the US.",
    "nine out of ten people will hate it",
    "The two stars of the film, Hanks and Ryan, are great.",
    "the 2 stars of this movie",
    "only 1/10 of the budget went to effects",
    "I'll give it a chance.",
    "give it a B movie treatment",
    "He gave it 2 hours of his life.",
    "Rated R for violence.",
    "It is a 1/5th scale model.",
    "In 1985/86 season",
    "He scored 10 goals",
    "11/10 would watch again",
    "Rated 4.5/10 of the book",
    "*** Rich and Strange (12/10/31)",
    "the rating 10 times higher",
    "Grade 5 art class",
]

def test_positive():
    for text, name in POSITIVE.items():
        ms = find_ratings(text)
        assert ms, f"no match: {text!r}"
        assert ms[0].pattern == name, f"{text!r}: got {ms[0].pattern}, want {name}"

def test_negative():
    for text in NEGATIVE:
        assert not has_rating(text), f"false positive: {text!r} -> {[m.text for m in find_ratings(text)]}"

def test_scrub():
    assert scrub("A movie of its time, and place. 2/10.") == "A movie of its time, and place."
    assert scrub("Fun (7/10) but long.") == "Fun but long."
    assert scrub("My rating: 8/10") == ""
    assert scrub("no rating here 9/11") == "no rating here 9/11"
    for text in POSITIVE:
        assert not has_rating(scrub(text)), text

def test_invert_and_score():
    assert invert_x10("Bad. 2/10") == "Bad. 8/10"
    assert invert_x10("seven out of ten") == "three out of ten"
    assert invert_x10("7.5/10 and 3 stars") == "2.5/10 and 3 stars"
    assert last_x10_score("first 3/10 then 8 out of 10") == 8.0
    assert last_x10_score("3 stars") is None

def test_html():
    assert clean_html("a<br /><br />b") == "a  b"

if __name__ == "__main__":
    for f in [test_positive, test_negative, test_scrub, test_invert_and_score, test_html]:
        f(); print("ok", f.__name__)
