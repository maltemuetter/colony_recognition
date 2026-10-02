from colony_recognition import ColonyCounter, CorrectColonyCount


counter = ColonyCounter("background.png")
recognition = counter.analyze_section("test.png", 2)

correction = CorrectColonyCount(recognition, "test")
correction.manual_correct()
