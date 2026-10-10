# PostgreSQL 15 image with the full benchmark database baked in, so it starts instantly
# with no load step:   docker run -d -p 127.0.0.1:5440:5432 vigneshwarr/chicago-crime-db
#
# This file documents how the published image was built. The data snapshot (gzip'd SQL parts
# in ./snapshot/) is not stored in git; it is only needed to rebuild the image:
#   docker build -f docker/Dockerfile -t vigneshwarr/chicago-crime-db:latest .

# ---- stage 1: initialise a cluster and load the snapshot --------------------------------------
FROM postgres:15 AS loader
# Not /var/lib/postgresql/data: the base image declares that path a VOLUME, so anything
# written there at build time would be discarded.
ENV PGDATA=/baked/pgdata
COPY snapshot/ /snapshot/
COPY docker/build_database.sh /build_database.sh
RUN bash /build_database.sh

# ---- stage 2: the runtime image holds only the finished data directory -------------------------
FROM postgres:15
ENV PGDATA=/baked/pgdata
COPY --from=loader --chown=postgres:postgres /baked/pgdata /baked/pgdata
LABEL org.opencontainers.image.title="chicago-crime-db" \
      org.opencontainers.image.description="PostgreSQL 15 with the Chicago crime benchmark database (crime_incidents + district/community-area/ward lookup tables), read-only role ai_agent"
