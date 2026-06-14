import { api } from "@/lib/api/client";
import type { QueueEntryWithItem, QueueItemType } from "@/types";

export const queuesApi = {
  list: () => api.get<QueueEntryWithItem[]>("/queues/"),
  moveToTop: (itemType: QueueItemType, itemId: number) => api.post(`/queues/${itemType}/${itemId}/move-to-top`),
};
