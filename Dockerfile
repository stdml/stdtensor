#!/usr/bin/env -S sh -c 'docker build --rm -t stdml/stdtensor:latest .'

FROM ubuntu:focal

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install -y build-essential cmake

WORKDIR /src
ADD . .

# CPackDeb: Debian package versioning ([<epoch>:]<version>[-<release>])
# should confirm to "^([0-9]+:)?[0-9][A-Za-z0-9.+~-]*$"
RUN ./configure --release=0-latest --deb && make && make package
