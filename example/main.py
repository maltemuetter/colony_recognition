from colony_counting import ColonyCounter, CorrectSection


counter = ColonyCounter("background.png")
section = counter.analyze_section("test.png", 2)

CorrectSection(section, "test")
