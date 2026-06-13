import { api } from "@/lib/api/client";
import type { QueueEntryWithJob } from "@/types";

export const queuesApi = {
  list: () => api.get<QueueEntryWithJob[]>("/queues/"),
  moveToTop: (jobId: number) => api.post(`/queues/${jobId}/move-to-top`),
};
