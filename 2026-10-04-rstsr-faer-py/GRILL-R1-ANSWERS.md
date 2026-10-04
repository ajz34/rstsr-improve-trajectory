Fact base: This repo claims most support but not full. You see there is table showing what we are lacking of.

Q1/Q10: Well you see if our tones converges. Phase 1/2 is something that need to do in rust-side code. And for this rstsr-faer-py task, you are not going to implement (may be fix if problems are dumb, but only very little) in rust side code. Virtually you are not going to edit rstsr-core. Your task is only implement a numpy-like library, or even just compatible to python-array-api, to make the array api test on this.

Q3: Well since we have rstsr-cpu-dlpack, maybe dlpack functionalities can be considered as that we have ways to implemented, not the current document states.

Q5: Okay. And also note, you are not going to really implement the python distribution package (to pypi or conda), but you are going to prepared to that.

Q7: Well if any repos that you need, clone to ~/Git-Others. Make rstsr itself somehow clean.

Q9: Emmm you are not going to CI. Proceed to CI until all rust-side code finished, which requires future work.

Q2/4/6/8: Okay.
