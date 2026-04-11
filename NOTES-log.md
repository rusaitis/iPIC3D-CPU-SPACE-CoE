# 04-09-2026 (Danny)
Here is a snippet of code from my heat conduction work that creates an event in -log_view for PETSc output. The standard output tracks stuff like SNESSolve, KSPSolve, etc. but you can add your own code using this syntax. 

PetscLogEvent plotTempEvent;
PetscCall(PetscLogEventRegister("PlotTemperature2D", PETSC_VIEWER_CLASSID, &plotTempEvent));
PetscCall(PetscLogEventBegin(plotTempEvent, 0, 0, 0, 0));
*CODE*
PetscCall(PetscLogEventEnd(plotTempEvent, 0, 0, 0, 0));

Like I said, I haven't done this with code outside of PETSc but I don't see why it shouldn't work. Also here is more information on profiling in PETSc (https://petsc.org/release/manual/profiling/). You can make flamegraphs using codes in the PETSc directory which can be useful visualizations.

