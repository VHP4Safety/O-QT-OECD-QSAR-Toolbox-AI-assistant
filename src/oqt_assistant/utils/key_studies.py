class KeyStudyCollector:
    def __init__(self, api):
        self.api = api
        self._about_cache = {}
        self._db_catalog_cache = {}

    def _about(self, guid):
        if not guid:
            return {}
        if guid not in self._about_cache:
            self._about_cache[guid] = self.api.get_object_about(guid) or {}
        return self._about_cache[guid]

    def _db_catalog(self, position, endpoint):
        k = (position or "", endpoint or "")
        if k not in self._db_catalog_cache:
            self._db_catalog_cache[k] = self.api.get_database_catalog(position, endpoint) or []
        return self._db_catalog_cache[k]

    def enrich_experimental_records(self, records):
        """Attach source caption + about (authors/url) when resolvable."""
        out = []
        for r in records or []:
            rp, ep = r.get("RigidPath"), r.get("Endpoint")
            # map SourceId->Guid/Caption via data/databases (best effort)
            dbs = self._db_catalog(rp, ep)
            source_caption, source_guid = None, None
            src_id = str(r.get("SourceId") or "")
            for db in dbs:
                # Toolbox often uses numeric SourceId with a DB Guid; we expose both if available
                if str(db.get("SourceId")) == src_id or not src_id:
                    source_caption = db.get("Caption") or source_caption
                    source_guid = db.get("Guid") or source_guid
            about = self._about(source_guid) if source_guid else {}
            r["Provenance"] = {
                "SourceCaption": source_caption,
                "SourceGuid": source_guid,
                "Donator": about.get("Donator"),
                "Authors": about.get("Authors"),
                "Url": about.get("Url"),
                "Disclaimer": about.get("Disclaimer")
            }
            out.append(r)
        return out