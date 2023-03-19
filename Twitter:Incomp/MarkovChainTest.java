package org.cis1200;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;

/** Tests for MarkovChain */
public class MarkovChainTest {

    /*
     * Writing tests for Markov Chain can be a little tricky.
     * We provide a few tests below to help you out, but you still need
     * to write your own.
     */

    /* **** ****** **** **** ADD BIGRAMS TESTS **** **** ****** **** */

    /* Here's an example test case. Be sure to add your own as well */
    @Test
    public void testAddBigram() {
        MarkovChain mc = new MarkovChain();
        mc.addBigram("1", "2");
        assertTrue(mc.chain.containsKey("1"));
        ProbabilityDistribution<String> pd = mc.chain.get("1");
        assertTrue(pd.getRecords().containsKey("2"));
        assertEquals(1, pd.count("2"));
    }

    @Test
    public void testAddBigramNullInput() {
        MarkovChain mc = new MarkovChain();
        try {
            mc.addBigram(null, "2");
        } catch (IllegalArgumentException e) {
            return;
        }
    }

    @Test
    public void testAddBigramAlreadyExists() {
        MarkovChain mc = new MarkovChain();
        mc.addBigram("1", "2");
        assertTrue(mc.chain.containsKey("1"));
        ProbabilityDistribution<String> pd = mc.chain.get("1");
        assertTrue(pd.getRecords().containsKey("2"));
        assertEquals(1, pd.count("2"));

        mc.addBigram("1", "2");
        ProbabilityDistribution<String> pd2 = mc.chain.get("1");
        assertTrue(pd2.getRecords().containsKey("2"));
        assertEquals(2, pd.count("2"));
    }

    /* ***** ****** ***** ***** TRAIN TESTS ***** ***** ****** ***** */

    /* Here's an example test case. Be sure to add your own as well */
    @Test
    public void testTrain() {
        MarkovChain mc = new MarkovChain();
        String sentence = "1 2 3";
        mc.train(Arrays.stream(sentence.split(" ")).iterator());
        assertEquals(3, mc.chain.size());
        ProbabilityDistribution<String> pd1 = mc.chain.get("1");
        assertTrue(pd1.getRecords().containsKey("2"));
        assertEquals(1, pd1.count("2"));
        ProbabilityDistribution<String> pd2 = mc.chain.get("2");
        assertTrue(pd2.getRecords().containsKey("3"));
        assertEquals(1, pd2.count("3"));
        ProbabilityDistribution<String> pd3 = mc.chain.get("3");
        assertTrue(pd3.getRecords().containsKey(MarkovChain.END_TOKEN));
        assertEquals(1, pd3.count(MarkovChain.END_TOKEN));
    }

    @Test
    public void testTrain2() {
        MarkovChain mc = new MarkovChain();
        String sentence = "How are you doing";
        mc.train(Arrays.stream(sentence.split(" ")).iterator());
        assertEquals(4, mc.chain.size());
        ProbabilityDistribution<String> pd1 = mc.chain.get("How");
        assertTrue(pd1.getRecords().containsKey("are"));
        assertTrue(!pd1.getRecords().containsKey("you"));
        assertEquals(1, pd1.count("are"));
        ProbabilityDistribution<String> pd2 = mc.chain.get("you");
        assertTrue(pd2.getRecords().containsKey("doing"));
        assertEquals(1, pd2.count("doing"));
    }

    @Test
    public void testAddedStartWord() {
        MarkovChain mc = new MarkovChain();
        String sentence = "How are you doing";
        mc.train(Arrays.stream(sentence.split(" ")).iterator());
        mc.startWords.keySet().contains("How");
    }


    /* **** ****** ****** MARKOV CHAIN CLASS TESTS ***** ****** ***** */

    /* Here's an example test case. Be sure to add your own as well */
    @Test
    public void testWalk() {
        /*
         * Using the sentences "CIS 120 rocks" and "CIS 120 beats CIS 160",
         * we're going to put some bigrams into the Markov Chain.
         *
         * While in the real world, we want the sentence we output to be random,
         * we don't want this in testing. For testing, we want to modify our
         * ProbabilityDistribution such that it will output a predictable chain
         * of words.
         *
         * Luckily, we've provided a `fixDistribution` method that will do this
         * for you! By calling `fixDistribution` with a list of words that you
         * expect to be output, the ProbabilityDistributions will be modified to
         * output your words in that order.
         *
         * See our below test for an example of how to use this.
         */

        String[] expectedWords = { "CIS", "120", "beats", "CIS", "120", "rocks" };
        MarkovChain mc = new MarkovChain();

        String sentence1 = "CIS 120 rocks";
        String sentence2 = "CIS 120 beats CIS 160";
        mc.train(Arrays.stream(sentence1.split(" ")).iterator());
        mc.train(Arrays.stream(sentence2.split(" ")).iterator());

        mc.reset("CIS"); // we start with "CIS" since that's the word our desired walk starts with
        mc.fixDistribution(new ArrayList<>(Arrays.asList(expectedWords)));

        for (int i = 0; i < expectedWords.length; i++) {
            assertTrue(mc.hasNext());
            assertEquals(expectedWords[i], mc.next());
        }
    }

    @Test
    public void testAnotherWalk() {

        String[] expectedWords = { "apples", "are", "good"};
        MarkovChain mc = new MarkovChain();

        String sentence1 = "apples are good";
        mc.train(Arrays.stream(sentence1.split(" ")).iterator());

        mc.reset("apples");

        for (int i = 0; i < expectedWords.length; i++) {
            assertTrue(mc.hasNext());
            assertEquals(expectedWords[i], mc.next());
        }
    }

    @Test
    public void testWalkResetTwice() {

        String[] expectedWords = { "apples", "are", "good", "bananas", "are", "better" };
        MarkovChain mc = new MarkovChain();

        String sentence1 = "apples are good";
        String sentence2 = "bananas are better";
        mc.train(Arrays.stream(sentence1.split(" ")).iterator());
        mc.train(Arrays.stream(sentence2.split(" ")).iterator());

        mc.reset("apples");

        assertTrue(mc.hasNext());
        assertEquals("apples", mc.next());

        mc.reset("bananas");

        assertTrue(mc.hasNext());
        assertEquals("bananas", mc.next());
    }
}
