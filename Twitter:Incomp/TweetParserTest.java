package org.cis1200;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import java.io.BufferedReader;
import java.io.StringReader;
import java.util.LinkedList;
import java.util.List;

/** Tests for TweetParser */
public class TweetParserTest {

    // A helper function to create a singleton list from a word
    private static List<String> singleton(String word) {
        List<String> l = new LinkedList<String>();
        l.add(word);
        return l;
    }

    // A helper function for creating lists of strings
    private static List<String> listOfArray(String[] words) {
        List<String> l = new LinkedList<String>();
        for (String s : words) {
            l.add(s);
        }
        return l;
    }

    // Cleaning and filtering tests -------------------------------------------
    @Test
    public void removeURLsTest() {
        assertEquals("abc . def.", TweetParser.removeURLs("abc http://www.cis.upenn.edu. def."));
        assertEquals("abc", TweetParser.removeURLs("abc"));
        assertEquals("abc ", TweetParser.removeURLs("abc http://www.cis.upenn.edu"));
        assertEquals("abc .", TweetParser.removeURLs("abc http://www.cis.upenn.edu."));
        assertEquals(" abc ", TweetParser.removeURLs("http:// abc http:ala34?#?"));
        assertEquals(" abc  def", TweetParser.removeURLs("http:// abc http:ala34?#? def"));
        assertEquals(" abc  def", TweetParser.removeURLs("https:// abc https``\":ala34?#? def"));
        assertEquals("abchttp", TweetParser.removeURLs("abchttp"));
    }

    @Test
    public void testCleanWord() {
        assertEquals("abc", TweetParser.cleanWord("abc"));
        assertEquals("abc", TweetParser.cleanWord("ABC"));
        assertNull(TweetParser.cleanWord("@abc"));
        assertEquals("ab'c", TweetParser.cleanWord("ab'c"));
    }

    /* **** ****** **** WRITE YOUR TESTS BELOW THIS LINE **** ****** **** */

    @Test
    public void testCleanWordAgain() {
        assertEquals("what", TweetParser.cleanWord("wHAt"));
        assertEquals("it's", TweetParser.cleanWord("it's"));
        assertNull(TweetParser.cleanWord("@jana"));
    }

    /* **** ****** ***** **** EXTRACT COLUMN TESTS **** **** ****** ***** */

    /* Here's an example test case. Be sure to add your own as well */
    @Test
    public void testExtractColumnGetsCorrectColumn() {
        assertEquals(
                " This is a tweet.",
                TweetParser.extractColumn(
                        "wrongColumn, wrong column, wrong column!, This is a tweet.", 3
                )
        );
    }

    @Test
    public void testExtractColumnFirstColumn() {
        assertEquals(
                "This one!",
                TweetParser.extractColumn(
                        "This one!, wrong column, wrong column.", 0
                )
        );
    }

    @Test
    public void testExtractColumnColumnOutOfBounds() {
        assertEquals(
                null,
                TweetParser.extractColumn(
                        "wrongColumn, wrong column, wrong column!, This is a tweet.", 5
                )
        );
    }

    @Test
    public void testExtractColumnStringNull() {
        assertEquals(
                null,
                TweetParser.extractColumn(
                        null, 0
                )
        );
    }

    /* **** ****** ***** ***** CSV DATA TO TWEETS ***** **** ****** ***** */

    /* Here's an example test case. Be sure to add your own as well */
    @Test
    public void testCsvDataToTweetsSimpleCSV() {
        StringReader sr = new StringReader(
                "0, The end should come here.\n" +
                        "1, This comes from data with no duplicate words!"
        );
        BufferedReader br = new BufferedReader(sr);
        List<String> tweets = TweetParser.csvDataToTweets(br, 1);
        List<String> expected = new LinkedList<String>();
        expected.add(" The end should come here.");
        expected.add(" This comes from data with no duplicate words!");
        assertEquals(expected, tweets);
    }

    @Test
    public void testCsvDataToTweetsSimpleCSV2() {
        StringReader sr = new StringReader(
                "0, The end should come here.\n" +
                        "1, This comes from data with no duplicate words!\n"
                            + "2, This should be included too."
        );
        BufferedReader br = new BufferedReader(sr);
        List<String> tweets = TweetParser.csvDataToTweets(br, 1);
        List<String> expected = new LinkedList<String>();
        expected.add(" The end should come here.");
        expected.add(" This comes from data with no duplicate words!");
        expected.add(" This should be included too.");
        assertEquals(expected, tweets);
    }

    @Test
    public void testCsvDataToTweetsNumbers() {
        StringReader sr = new StringReader(
                "0, The end should come here.\n" +
                        "1, This comes from data with no duplicate words!\n"
                        + "2, This should be included too."
        );
        BufferedReader br = new BufferedReader(sr);
        List<String> tweets = TweetParser.csvDataToTweets(br, 0);
        List<String> expected = new LinkedList<String>();
        expected.add("0");
        expected.add("1");
        expected.add("2");
        assertEquals(expected, tweets);
    }

    @Test
    public void testCsvDataToTweetsExtraColumn() {
        StringReader sr = new StringReader(
                "0, The end should come here., Hi\n" +
                        "1, This comes from data with no duplicate words!, Hi\n"
                        + "2, This should be included too., Hi"
        );
        BufferedReader br = new BufferedReader(sr);
        List<String> tweets = TweetParser.csvDataToTweets(br, 2);
        List<String> expected = new LinkedList<String>();
        expected.add(" Hi");
        expected.add(" Hi");
        expected.add(" Hi");
        assertEquals(expected, tweets);
    }

    /* **** ****** ***** ** PARSE AND CLEAN SENTENCE ** ***** ****** ***** */

    /* Here's an example test case. Be sure to add your own as well */
    @Test
    public void parseAndCleanSentenceNonEmptyFiltered() {
        List<String> sentence = TweetParser.parseAndCleanSentence("abc #@#F");
        List<String> expected = new LinkedList<String>();
        expected.add("abc");
        assertEquals(expected, sentence);
    }

    @Test
    public void parseAndCleanSentenceMultWords() {
        List<String> sentence = TweetParser.parseAndCleanSentence("abc ADC @hi Why");
        List<String> expected = new LinkedList<String>();
        expected.add("abc");
        expected.add("adc");
        expected.add("why");
        assertEquals(expected, sentence);
    }

    @Test
    public void parseAndCleanSentenceAllBad() {
        List<String> sentence = TweetParser.parseAndCleanSentence("#helo #@#F @no");
        List<String> expected = new LinkedList<String>();
        assertEquals(expected, sentence);
    }

    /* **** ****** ***** **** PARSE AND CLEAN TWEET *** ***** ****** ***** */

    /* Here's an example test case. Be sure to add your own as well */
    @Test
    public void testParseAndCleanTweetRemovesURLS1() {
        List<List<String>> sentences = TweetParser
                .parseAndCleanTweet("abc http://www.cis.upenn.edu");
        List<List<String>> expected = new LinkedList<List<String>>();
        expected.add(singleton("abc"));
        assertEquals(expected, sentences);
    }

    @Test
    public void testParseAndCleanTweetTwoSentences() {
        List<List<String>> sentences = TweetParser
                .parseAndCleanTweet("abc http://www.cis.upenn.edu. This is another SENTENCE.");
        List<List<String>> expected = new LinkedList<List<String>>();
        expected.add(singleton("abc"));
        expected.add(listOfArray("this is another sentence".split(" ")));
        assertEquals(expected, sentences);
    }

    @Test
    public void testParseAndCleanIncludesBad() {
        List<List<String>> sentences = TweetParser
                .parseAndCleanTweet("abc http://www.cis.upenn.edu. @jana how are you.");
        List<List<String>> expected = new LinkedList<List<String>>();
        expected.add(singleton("abc"));
        expected.add(listOfArray("how are you".split(" ")));
        assertEquals(expected, sentences);
    }

    /* **** ****** ***** ** CSV DATA TO TRAINING DATA ** ***** ****** **** */

    /* Here's an example test case. Be sure to add your own as well */
    @Test
    public void testCsvDataToTrainingDataSimpleCSV() {
        StringReader sr = new StringReader(
                "0, The end should come here.\n" +
                        "1, This comes from data with no duplicate words!"
        );
        BufferedReader br = new BufferedReader(sr);
        List<List<String>> tweets = TweetParser.csvDataToTrainingData(br, 1);
        List<List<String>> expected = new LinkedList<List<String>>();
        expected.add(listOfArray("the end should come here".split(" ")));
        expected.add(listOfArray("this comes from data with no duplicate words".split(" ")));
        assertEquals(expected, tweets);
    }

    @Test
    public void testCsvDataToTrainingDataMoreColumns() {
        StringReader sr = new StringReader(
                "0, The end should come here., Another column. Another sentence.\n" +
                        "1, This comes from data with no duplicate words!, " +
                        "YET another column. YET another sentence."
        );
        BufferedReader br = new BufferedReader(sr);
        List<List<String>> tweets = TweetParser.csvDataToTrainingData(br, 2);
        List<List<String>> expected = new LinkedList<List<String>>();
        expected.add(listOfArray("another column".split(" ")));
        expected.add(listOfArray("another sentence".split(" ")));
        expected.add(listOfArray("yet another column".split(" ")));
        expected.add(listOfArray("yet another sentence".split(" ")));
        assertEquals(expected, tweets);
    }

    @Test
    public void testCsvDataToTrainingDataIncludeBad() {
        StringReader sr = new StringReader(
                "0, The end should come here @cis1200.\n" +
                        "1, This comes from data with no duplicate words " +
                        "https://www.seas.upenn.edu/~cis120/22fa/schedule/!"
        );
        BufferedReader br = new BufferedReader(sr);
        List<List<String>> tweets = TweetParser.csvDataToTrainingData(br, 1);
        List<List<String>> expected = new LinkedList<List<String>>();
        expected.add(listOfArray("the end should come here".split(" ")));
        expected.add(listOfArray("this comes from data with no duplicate words".split(" ")));
        assertEquals(expected, tweets);
    }

}
