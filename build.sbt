organization  := "ch.unibas.cs.gravis"

name := """scalismo-tutorial"""
version       := "0.16.0"

scalaVersion  := "2.12.8"

scalacOptions := Seq("-unchecked", "-deprecation", "-encoding", "utf8")

resolvers += Resolver.bintrayRepo("unibas-gravis", "maven")

resolvers += Opts.resolver.sonatypeSnapshots


libraryDependencies  ++= Seq(
            "ch.unibas.cs.gravis" % "scalismo-native-all" % "4.0.0",
            "ch.unibas.cs.gravis" %% "scalismo-ui" % "0.12.1"
)

lazy val root = (project in file("."))

lazy val docs = project       // new documentation project
  .dependsOn(root)
  .enablePlugins(MdocPlugin)
 