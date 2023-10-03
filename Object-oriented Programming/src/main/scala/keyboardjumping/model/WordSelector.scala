package keyboardjumping.model

import scala.collection.mutable.ListBuffer
import scala.util.Random

class WordSelector {
  var words = ListBuffer("resilient", "acquiesce", "plethora", "zealous", "nostalgia", "paradox", "vicarious", "querulous", "vexation", "audacity", "furtive", "haughty", "implorer", "narcissi", "pensive", "scuttle", "tumult", "wheedle", "bucolic", "conflate", "devious", "eloquent", "ferocity", "gluttony", "heedless", "incisive", "freedom", "inspire", "journey", "laughter", "melodic", "passion", "rapture", "serenity", "sunset", "triumph", "whimsical", "zenith")

  def getRandomWord(): String = {
    val randomIndex = Random.nextInt(words.length)
    val word = words(randomIndex)
    words.remove(randomIndex)
    word
  }

  def resetWords = words = ListBuffer("resilient", "acquiesce", "plethora", "zealous", "nostalgia", "paradox", "vicarious", "querulous", "vexation", "audacity", "furtive", "haughty", "implorer", "narcissi", "pensive", "scuttle", "tumult", "wheedle", "bucolic", "conflate", "devious", "eloquent", "ferocity", "gluttony", "heedless", "incisive", "freedom", "inspire", "journey", "laughter", "melodic", "passion", "rapture", "serenity", "sunset", "triumph", "whimsical", "zenith")
}
